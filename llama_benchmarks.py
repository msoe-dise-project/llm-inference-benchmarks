#!/usr/bin/env python3
"""
OpenAI REST API Benchmark Tool

Benchmarks concurrent REST requests to an Open-AI compatible server and records timing data.
"""

import asyncio
import aiohttp
import argparse
import csv
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
import random

@dataclass
class RequestResult:
    """Results from a single request"""
    request_id: int
    start_time: float
    end_time: float
    duration: float
    status_code: int
    success: bool
    prompt: str
    prompt_tokens: 0
    response_tokens: 0
    error_message: str = ""
    response: str = ""

class OllamaBenchmark:
    def __init__(self, base_url: str, prompts: list[str]):
        self.base_url = base_url.rstrip('/')
        self.results: List[RequestResult] = []
        self.prompts = prompts
    
    async def make_request(self, session: aiohttp.ClientSession, request_id: int, 
                          model: str) -> RequestResult:
        """Make a single request to API"""
        prompt = random.choice(self.prompts)
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
              {
                "role" : "user",
                "content" : prompt
              }  
            ],
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            async with session.post(url, json=payload) as response:
                resp = await response.text()  # Read response body
                end_time = time.time()
                duration = end_time - start_time

                resp_json = json.loads(resp)

                print(".", end="", flush=True)

                return RequestResult(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    status_code=response.status,
                    success=response.status == 200,
                    prompt=prompt,
                    prompt_tokens=resp_json["usage"]["prompt_tokens"],
                    response_tokens=resp_json["usage"]["completion_tokens"],
                    response=resp,
                )
        except Exception as e:
            end_time = time.time()
            return RequestResult(
                request_id=request_id,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                status_code=0,
                success=False,
                prompt=prompt,
                prompt_tokens=0,
                response_tokens=0,
                error_message=str(e)
            )
    
    async def run_benchmark(self, num_requests: int, concurrency: int, 
                           model: str):
        """Run benchmark with specified concurrency"""
        connector = aiohttp.TCPConnector(limit=concurrency)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Warm up runner
            #print("Running single request to warm up runner...")
            #response = await self.make_request(session, -1, model)
            #print()
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def bounded_request(request_id: int):
                async with semaphore:
                    return await self.make_request(session, request_id, model)
            
            # Create tasks for all requests
            tasks = [bounded_request(i) for i in range(num_requests)]
            
            # Execute all requests
            print(f"Starting {num_requests} requests with concurrency {concurrency}...")
            start_benchmark = time.time()
            
            self.results = await asyncio.gather(*tasks)
            print()
            
            end_benchmark = time.time()
            elapsed = end_benchmark - start_benchmark
            return elapsed
        
    def print_summary(self, elapsed, n_requests):
        """Print benchmark summary statistics"""
        if not self.results:
            print("No results to summarize")
            return
        
        successful_requests = [r for r in self.results if r.success]
        failed_requests = [r for r in self.results if not r.success]
        
        if successful_requests:
            durations = [r.duration for r in successful_requests]
            avg_duration = sum(durations) / len(durations)
            std_duration = np.std(durations)

            prompt_tokens = [r.prompt_tokens for r in successful_requests]
            avg_prompt_tokens = np.mean(prompt_tokens)
            std_prompt_tokens = np.std(prompt_tokens)

            completion_tokens = [r.response_tokens for r in successful_requests]
            avg_tokens = np.mean(completion_tokens)
            std_tokens = np.std(completion_tokens)

            tokens_per_sec = [r.response_tokens / r.duration for r in successful_requests]
            avg_tokens_per_sec = np.mean(tokens_per_sec)
            std_tokens_per_sec = np.std(tokens_per_sec)

            # parallel_tokens_per_sec = np.sum(completion_tokens) / elapsed

            avg_sec_per_request = elapsed / n_requests

            tokens_q50, tokens_q10, tokens_q05 = np.quantile(tokens_per_sec, [0.5, 0.1, 0.05])
            response_q50, response_q90, response_q95 = np.quantile(durations, [0.5, 0.9, 0.95])

            print(f"\n=== Benchmark Summary ===")
            print(f"Total requests: {len(self.results)}")
            print(f"Successful: {len(successful_requests)}")
            print(f"Failed: {len(failed_requests)}")
            print(f"Success rate: {len(successful_requests) / len(self.results) * 100:.1f}%")
            print()
            print(f"Average number of prompt tokens: {avg_prompt_tokens:.1f}")
            print(f"Std dev number of prompt tokens: {std_prompt_tokens:.1f}")
            print()
            # this measures throughput
            print(f"Benchmark elapsed time: {elapsed:.1f}s")
            # this should go over as concurrency increases
#            print(f"Average response time: {avg_sec_per_request:.1f}s")
            # hopefully without hurting this
            print(f"Sequential response time: {avg_duration:.1f}s, {response_q90:.1f}s, {response_q95:.1f}s")
#            print(f"Standard dev seq response time: {std_duration:.1f}s")
            print()
            print(f"Average number of generated tokens: {avg_tokens:.1f}")
            print(f"Std dev number of generated tokens: {std_tokens:.1f}")
            print()
            # this measures throughput
            # this should go over as concurrency increases
#            print(f"Average tokens generated / sec across all requests: {tokens_per_sec:.1f}")
            # hopefully without hurting this
            print(f"Tokens generated / request / sec: {tokens_q05:.1f}, {tokens_q10:.1f}, {avg_tokens_per_sec:.1f}")
#            print(f"Std dev tokens generated / request / sec: {std_tokens_per_sec:.1f}")
            print()

        else:
            print(f"All {len(self.results)} requests failed")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp REST API")
    parser.add_argument("--url", required=True, 
                       help="llama.cpp server URL")
    parser.add_argument("--requests", "-r", type=int, required=True,
                       help="Number of requests to make")
    parser.add_argument("--concurrency", "-c", type=int, required=True,
                       help="Number of concurrent requests")
    parser.add_argument("--prompt-file", "-p", type=str, required=True,
                       help="Prompt file to use")
    parser.add_argument("--model", "-m", required=True,
                       help="Model to use")    

    args = parser.parse_args()

    prompts_df = pd.read_csv(args.prompt_file)
    prompts = list(prompts_df["Instruction"])
    
    benchmark = OllamaBenchmark(args.url, prompts)
    
    elapsed = await benchmark.run_benchmark(
        num_requests=args.requests,
        concurrency=args.concurrency,
        model=args.model
    )
    
    benchmark.print_summary(elapsed, args.requests)
        

if __name__ == "__main__":
    asyncio.run(main())
