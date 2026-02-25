import os
import json
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any, Optional

import requests

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # optional fallback


# 讀取 prompts.json，支援以下格式：
# - ["p1", "p2", ...]
# - {"prompts": ["p1", ...]}
# - {"items": [{"prompt": "..."}, ...]}

def _load_prompts(path: str, n: int) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[str] = []
    if isinstance(data, dict):
        if isinstance(data.get("prompts"), list):
            prompts = [x for x in data["prompts"] if isinstance(x, str)]
        else:
            for v in data.values():
                if isinstance(v, list):
                    if v and all(isinstance(x, str) for x in v):
                        prompts = v
                        break
                    cand = [x.get("prompt") for x in v if isinstance(x, dict) and "prompt" in x]
                    cand = [x for x in cand if isinstance(x, str)]
                    if cand:
                        prompts = cand
                        break
    elif isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            prompts = data
        else:
            prompts = [x.get("prompt") for x in data if isinstance(x, dict) and "prompt" in x]
            prompts = [x for x in prompts if isinstance(x, str)]

    prompts = [p for p in prompts if isinstance(p, str)]
    if not prompts:
        raise ValueError("prompts.json 內沒有可用的字串提示")
    if len(prompts) < n:
        times = (n + len(prompts) - 1) // len(prompts)
        prompts = (prompts * max(1, times))[:n]
    else:
        prompts = prompts[:n]
    return prompts


def _percentiles(values: List[float], ps=(50, 90, 95, 99)) -> List[Tuple[int, float]]:
    if not values:
        return [(p, math.nan) for p in ps]
    arr = sorted(values)
    out: List[Tuple[int, float]] = []
    for p in ps:
        k = (len(arr) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        v = arr[int(k)] if f == c else (arr[f] * (c - k) + arr[c] * (k - f))
        out.append((p, float(v)))
    return out


def _to_bool(s: Optional[str], default=False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1", "true", "t", "yes", "y"}


def _send_request(url: str, payload: dict, timeout_s: float) -> Tuple[bool, float, Any]:
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        dt = time.time() - t0
        try:
            return True, dt, resp.json()
        except ValueError:
            return True, dt, resp.text
    except Exception as e:  # pragma: no cover
        dt = time.time() - t0
        return False, dt, str(e)


def benchmark():
    # 基本參數（可用環境變數覆寫）
    host = os.environ.get("VLLM_HOST", os.environ.get("LL_HOST", "127.0.0.1"))
    port = int(os.environ.get("VLLM_PORT", os.environ.get("LL_PORT", "8000")))
    path = os.environ.get("VLLM_PATH", "/v1/chat/completions")
    prompts_json = os.environ.get("PROMPTS_JSON", os.path.join(os.path.dirname(__file__), "prompts.json"))
    n = int(os.environ.get("PROMPTS_N", "100"))
    concurrency = int(os.environ.get("CONCURRENCY", "8"))
    max_tokens = int(os.environ.get("MAX_TOKENS", os.environ.get("MAX_NEW_TOKENS", "256")))
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    timeout_s = float(os.environ.get("TIMEOUT_S", "300"))
    model_dir = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "model", "Llama-3-8B-Instruct"))

    url = f"http://{host}:{port}{path}"

    print(f"Target: {url}")
    print(f"Load prompts from: {prompts_json} (first {n})")

    prompts = _load_prompts(prompts_json, n)

    # 嘗試載入 tokenizer，若缺少用量時作為估算後備
    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        except Exception as e:  # pragma: no cover
            print(f"Warning: 無法載入 tokenizer（{e}），缺少 usage 時無法精準計算 token/s。")

    def _build_payload(p: str) -> dict:
        # 使用聊天 API 結構
        payload = {
            "messages": [
                {"role": "user", "content": p}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # 若伺服器支援，可開啟詳細資訊（非 OpenAI 規格，但部分實作支援）
        if _to_bool(os.environ.get("RETURN_DETAILS", "false")):
            payload["details"] = True
        return payload

    # 暖機
    ok, dt, _ = _send_request(url, _build_payload(prompts[0]), timeout_s)
    print(f"Warmup: ok={ok} latency={dt:.2f}s")

    latencies: List[float] = []
    errors: List[str] = []
    total_in_tokens = 0
    total_out_tokens = 0
    lock = threading.Lock()

    def _extract_generated_text(obj: Any) -> Optional[str]:
        try:
            if isinstance(obj, dict):
                # OpenAI 風格：choices[0].message.content
                if isinstance(obj.get("choices"), list) and obj["choices"] and isinstance(obj["choices"][0], dict):
                    ch = obj["choices"][0]
                    if isinstance(ch.get("message"), dict):
                        msg = ch["message"]
                        if isinstance(msg.get("content"), str):
                            return msg["content"]
                    # 也容忍 text 欄位
                    if isinstance(ch.get("text"), str):
                        return ch["text"]
                # 其他常見鍵
                if isinstance(obj.get("generated_text"), str):
                    return obj.get("generated_text")
                if isinstance(obj.get("text"), str):
                    return obj.get("text")
                # 某些實作可能將結果包在 data 中
                if isinstance(obj.get("data"), list) and obj["data"] and isinstance(obj["data"][0], dict):
                    return _extract_generated_text(obj["data"][0])
            elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return _extract_generated_text(obj[0])
        except Exception:  # pragma: no cover
            return None
        return None

    def _extract_usage(obj: Any, prompt_text: str) -> Tuple[int, int]:
        # 優先使用 usage；若缺失則用 tokenizer 估算
        try:
            if isinstance(obj, dict):
                if isinstance(obj.get("usage"), dict):
                    u = obj["usage"]
                    pt = int(u.get("prompt_tokens", 0))
                    ct = int(u.get("completion_tokens", 0))
                    if pt or ct:
                        return pt, ct
                # 其他常見鍵（LightLLM）
                pt2 = obj.get("prompt_tokens", None)
                ct2 = obj.get("count_output_tokens", None)
                if isinstance(pt2, int) and isinstance(ct2, int):
                    return pt2, ct2
                if isinstance(ct2, list) and ct2:
                    ct_sum = sum(int(x) for x in ct2 if isinstance(x, (int, float)))
                    pt_val = int(pt2) if isinstance(pt2, (int, float)) else 0
                    return pt_val, ct_sum
                # TGI 風格：details.generated_tokens / details.tokens / details.prefill
                if isinstance(obj.get("details"), dict):
                    det = obj["details"]
                    ct_det = int(det.get("generated_tokens", det.get("num_generated_tokens", 0)) or 0)
                    if not ct_det and isinstance(det.get("tokens"), list):
                        ct_det = len(det["tokens"])  # 非精準但可行
                    pt_det = 0
                    if isinstance(det.get("prefill"), list):
                        pt_det = len(det["prefill"])  # 近似輸入 token 數
                    if isinstance(det.get("usage"), dict):
                        du = det["usage"]
                        pt_det = int(du.get("prompt_tokens", pt_det))
                        ct_det = int(du.get("completion_tokens", ct_det))
                    if pt_det or ct_det:
                        return pt_det, ct_det
                # 其他命名：num_input_tokens / num_generated_tokens
                num_in = obj.get("num_input_tokens")
                num_out = obj.get("num_generated_tokens")
                if isinstance(num_in, (int, float)) or isinstance(num_out, (int, float)):
                    return int(num_in or 0), int(num_out or 0)
            elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                return _extract_usage(obj[0], prompt_text)
        except Exception:
            pass

        # 若無法解析，嘗試用 tokenizer 估算
        if tokenizer is None:
            return 0, 0
        try:
            p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            p_len = len(p_ids)
        except Exception:
            p_len = 0
        c_len = 0
        gen_text = _extract_generated_text(obj)
        if isinstance(gen_text, str):
            try:
                both = tokenizer(prompt_text + gen_text, add_special_tokens=False).input_ids
                c_len = max(0, len(both) - p_len)
            except Exception:
                try:
                    c_len = len(tokenizer(gen_text, add_special_tokens=False).input_ids)
                except Exception:
                    c_len = 0
        return p_len, c_len

    def _worker(p: str):
        succ, dt, obj = _send_request(url, _build_payload(p), timeout_s)
        with lock:
            latencies.append(dt)
        if not succ:
            with lock:
                errors.append(str(obj))
            return
        pt, ct = _extract_usage(obj, p)
        with lock:
            nonlocal total_in_tokens, total_out_tokens
            total_in_tokens += pt
            total_out_tokens += ct

    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_worker, p) for p in prompts]
        for _ in as_completed(futures):
            pass
    total_time = time.time() - start

    total = len(prompts)
    req_s = total / total_time if total_time > 0 else 0.0
    p_stats = _percentiles(latencies, ps=(50, 90, 95, 99))

    print("\n=== vLLM/Chat Completions Throughput Result ===")
    print(f"IP: {host}")
    print(f"Port: {port}")
    print(f"Endpoint: {path}")
    print(f"Requests: {total}")
    print(f"Concurrency: {concurrency}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput (requests): {req_s:.2f} req/s")
    if latencies:
        avg = sum(latencies) / len(latencies)
        print(f"Latency avg: {avg:.2f}s")
        print("Latency percentiles:")
        for p, v in p_stats:
            print(f"  p{p}: {v:.2f}s")

    if total_out_tokens > 0 or total_in_tokens > 0:
        tok_s_out = total_out_tokens / total_time if total_time > 0 else 0.0
        tok_s_in = total_in_tokens / total_time if total_time > 0 else 0.0
        tok_s_total = (total_in_tokens + total_out_tokens) / total_time if total_time > 0 else 0.0
        print(f"Throughput (output tokens): {tok_s_out:.2f} tok/s")
        print(f"Throughput (input tokens): {tok_s_in:.2f} tok/s")
        print(f"Throughput (total tokens): {tok_s_total:.2f} tok/s")

    if errors:
        print(f"Errors: {len(errors)} (showing first 3)")
        for e in errors[:3]:
            print(f"  - {e}")


if __name__ == "__main__":
    benchmark()
