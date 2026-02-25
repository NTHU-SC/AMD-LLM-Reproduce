import os
import json
import time
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Any

import requests
try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # optional fallback

# LL_HOST 預設 127.0.0.1
# LL_PORT 預設 8000
# PROMPTS_JSON 預設為與腳本同層的 prompts.json
# PROMPTS_N 預設 100
# CONCURRENCY 預設 8
# MAX_NEW_TOKENS 預設 128
# DO_SAMPLE 預設 false
# TIMEOUT_S 預設 300
# Run
# PROMPTS_N=10 LL_HOST=gn1228 LL_PORT=8080 python run_lightllm.py


def _to_bool(s: str, default=False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1", "true", "t", "yes", "y"}


def _percentiles(values: List[float], ps=(50, 90, 95, 99)) -> List[Tuple[int, float]]:
    if not values:
        return [(p, math.nan) for p in ps]
    arr = sorted(values)
    out = []
    for p in ps:
        k = (len(arr) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            v = arr[int(k)]
        else:
            v = arr[f] * (c - k) + arr[c] * (k - f)
        out.append((p, float(v)))
    return out


def load_prompts(path: str, n: int) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[str] = []
    if isinstance(data, dict):
        if "prompts" in data and isinstance(data["prompts"], list):
            prompts = data["prompts"]
        else:
            # 嘗試從第一個 list 值中讀取物件陣列的 prompt 欄位
            for v in data.values():
                if isinstance(v, list):
                    if v and all(isinstance(x, str) for x in v):
                        prompts = v
                        break
                    cand = [x.get("prompt") for x in v if isinstance(x, dict) and "prompt" in x]
                    if cand:
                        prompts = cand
                        break
    elif isinstance(data, list):
        # 支援純字串陣列或物件陣列
        if all(isinstance(x, str) for x in data):
            prompts = data
        else:
            prompts = [x.get("prompt") for x in data if isinstance(x, dict) and "prompt" in x]

    prompts = [p for p in prompts if isinstance(p, str)]
    if not prompts:
        raise ValueError("prompts.json 內沒有可用的字串提示")
    if len(prompts) < n:
        # 不足 n 筆時循環補齊
        times = (n + len(prompts) - 1) // len(prompts)
        prompts = (prompts * max(1, times))[:n]
    else:
        prompts = prompts[:n]
    return prompts


def send_request(url: str, prompt: str, params: dict, timeout_s: float) -> Tuple[bool, float, Any]:
    t0 = time.time()
    try:
        # TGI 的 details 應位於最上層，而非 parameters 內
        payload_params = dict(params) if isinstance(params, dict) else {}
        details_flag = payload_params.pop("details", None)
        body = {"inputs": prompt, "parameters": payload_params}
        if details_flag is not None:
            body["details"] = bool(details_flag)

        resp = requests.post(url, json=body, timeout=timeout_s)
        resp.raise_for_status()
        dt = time.time() - t0
        try:
            return True, dt, resp.json()
        except ValueError:
            return True, dt, resp.text
    except Exception as e:
        dt = time.time() - t0
        return False, dt, str(e)


def benchmark():
    # 可調參數（亦支援環境變數）
    host = os.environ.get("LL_HOST", "127.0.0.1")
    port = int(os.environ.get("LL_PORT", "8000"))
    url = f"http://{host}:{port}/generate"

    prompts_json = os.environ.get("PROMPTS_JSON", os.path.join(os.path.dirname(__file__), "prompts.json"))
    n = int(os.environ.get("PROMPTS_N", "100"))
    concurrency = int(os.environ.get("CONCURRENCY", "8"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))
    do_sample = _to_bool(os.environ.get("DO_SAMPLE", "false"))
    timeout_s = float(os.environ.get("TIMEOUT_S", "300"))
    model_dir = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "model", "Llama-3-8B-Instruct"))

    print(f"Target: {url}")
    print(f"Load prompts from: {prompts_json} (first {n})")

    prompts = load_prompts(prompts_json, n)
    params = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "details": True,
        # 要求 TGI 回傳輸入端細節（prefill），以便正確計算 input tokens
        "decoder_input_details": True,
    }

    # optional tokenizer for fallback token counting
    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        except Exception as e:
            print(f"Warning: 無法載入 tokenizer（{e}），將無法在缺少 usage 時精準計算 token/s。")

    # 暖機
    ok, dt, _ = send_request(url, prompts[0], {"max_new_tokens": 1, "do_sample": False}, timeout_s)
    print(f"Warmup: ok={ok} latency={dt:.2f}s")

    # 併發送出並統計
    latencies: List[float] = []
    errors: List[str] = []
    total_in_tokens = 0
    total_out_tokens = 0
    lock = threading.Lock()

    def _extract_token_usage(obj: Any, prompt_text: str) -> Tuple[int, int]:
        # return (prompt_tokens, completion_tokens)
        try:
            if isinstance(obj, dict):
                # OpenAI/vLLM 風格 usage
                usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
                if usage:
                    pt = usage.get("prompt_tokens")
                    ct = usage.get("completion_tokens")
                    if isinstance(pt, int) and isinstance(ct, int):
                        return pt, ct

                # LightLLM /generate 回傳：prompt_tokens 與 count_output_tokens
                pt = obj.get("prompt_tokens", None)
                ct = obj.get("count_output_tokens", None)
                if isinstance(pt, int) and isinstance(ct, int):
                    return pt, ct
                # 有時 generated_text 可能是 list 對應 n>1，count_output_tokens 也可能是 list
                if isinstance(ct, list) and ct:
                    ct_sum = sum(int(x) for x in ct if isinstance(x, (int, float)))
                    pt_val = int(pt) if isinstance(pt, (int, float)) else 0
                    return pt_val, ct_sum

                # TGI details（需帶 details=True）
                details = obj.get("details") if isinstance(obj.get("details"), dict) else None
                if details:
                    # 1) 優先使用伺服端直接提供的計數
                    pt = (
                        details.get("input_token_count")
                        or details.get("prompt_tokens")
                        or details.get("num_input_tokens")
                        or details.get("input_length")
                    )
                    ct = (
                        details.get("output_token_count")
                        or details.get("completion_tokens")
                        or details.get("num_generated_tokens")
                    )
                    if isinstance(pt, int) and isinstance(ct, int):
                        return pt, ct

                    # 2) 次選：generated_tokens（輸出），prefill 長度（輸入，含 special 以貼近伺服端計數）
                    prefill = details.get("prefill") or []
                    tokens = details.get("tokens") or []
                    pt2 = len(prefill) if isinstance(prefill, list) else None
                    ct2 = details.get("generated_tokens")
                    if not isinstance(ct2, int):
                        ct2 = len(tokens) if isinstance(tokens, list) else 0
                    if isinstance(pt2, int) and isinstance(ct2, int):
                        return pt2, ct2
            elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                # 容錯：若是 list 包裹單一物件
                return _extract_token_usage(obj[0], prompt_text)
        except Exception:
            pass

        # fallback: use tokenizer to estimate
        if tokenizer is None:
            return 0, 0
        try:
            p_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            p_len = len(p_ids)
        except Exception:
            p_len = 0

        gen_text = None
        if isinstance(obj, dict):
            gen_text = obj.get("generated_text") or obj.get("text")
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            gen_text = obj[0].get("generated_text") or obj[0].get("text")

        c_len = 0
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
        succ, dt, obj = send_request(url, p, params, timeout_s)
        with lock:
            latencies.append(dt)
        if not succ:
            with lock:
                errors.append(str(obj))
            return
        pt, ct = _extract_token_usage(obj, p)
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
    qps = total / total_time if total_time > 0 else 0.0
    p_stats = _percentiles(latencies, ps=(50, 90, 95, 99))

    print("\n=== Throughput Result ===")
    print(f"IP: {host}")
    print(f"Port: {port}")
    print(f"Requests: {total}")
    print(f"Concurrency: {concurrency}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {qps:.2f} req/s")
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

    # 額外輸出彙總 token 數
    print(f"\n[Stats] input_tokens={total_in_tokens}, generated_tokens={total_out_tokens}, total_tokens={total_in_tokens + total_out_tokens}")


if __name__ == "__main__":
    benchmark()