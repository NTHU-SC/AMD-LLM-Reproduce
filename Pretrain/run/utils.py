import time
import torch
import torch.distributed as dist

class BenchmarkTimer:
    def __init__(self, warm_up_steps=30, total_steps=100):
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps + warm_up_steps
        self.total_tokens = 0.0
        self.total_steps_time = 0.0
        self.start_time = None
        self.is_recording = False

    def step_start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    def step_end(self, start_ts, step, tokens_this_step):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time = time.time() - start_ts

        if step >= self.warm_up_steps:
            if not self.is_recording:
                self.is_recording = True
                self.start_time = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                print(f"\n[INFO] Warm-up finished at step {step}, starting measurement...")
            
            self.total_tokens += tokens_this_step
            self.total_steps_time += step_time
            
        status = "Recording" if self.is_recording else "Warm-up"
        return step_time, status

    def print_final_stats(self, rank=0):
        if rank == 0:
            if not self.is_recording or self.total_steps_time == 0:
                print("\n[Error] No data recorded. Ensure total_steps > warm_up_steps.")
                return

            avg_tps = self.total_tokens / self.total_steps_time
            total_elapsed = time.time() - self.start_time
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            print("\n" + "="*50)
            print(f"Final Benchmark Results (Excluding {self.warm_up_steps} warm-up steps)")
            print("-"*50)
            print(f"Avg Global Tokens/s:  {avg_tps:.2f}")
            print(f"Total Recording Time: {total_elapsed:.2f} s")
            print(f"Peak GPU Mem (GB):    {peak_mem:.2f}")
            print("="*50 + "\n")