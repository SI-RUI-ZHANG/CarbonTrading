"""Smart rate limiter with adaptive throttling and automatic recovery."""

import time
import threading
from collections import deque
from datetime import datetime, timedelta

class SmartRateLimiter:
    """Adaptive rate limiter with automatic recovery from 429 errors."""
    
    def __init__(self, calls_per_second: int = 50):
        self.base_rate = calls_per_second
        self.current_rate = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        
        # Sliding window for tracking calls
        self.call_times = deque(maxlen=calls_per_second * 60)  # 1 minute window
        self.lock = threading.Lock()
        
        # Track rate limit hits
        self.last_rate_limit_hit = None
        self.consecutive_successes = 0
        self.is_throttled = False
        
        # Statistics
        self.total_calls = 0
        self.total_rate_limit_hits = 0
        self.total_wait_time = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit."""
        with self.lock:
            current_time = time.time()
            
            # Clean old entries (older than 60 seconds)
            cutoff = current_time - 60
            while self.call_times and self.call_times[0] < cutoff:
                self.call_times.popleft()
            
            # Check if we're approaching the limit
            calls_in_window = len(self.call_times)
            max_calls_per_minute = self.current_rate * 60
            
            if calls_in_window >= max_calls_per_minute * 0.9:  # 90% threshold
                # Calculate how long to wait
                oldest_relevant = self.call_times[0]
                time_elapsed = current_time - oldest_relevant
                wait_time = 60.01 - time_elapsed  # Add small buffer
                
                if wait_time > 0:
                    print(f"  ⏳ Rate limit approaching ({calls_in_window}/{int(max_calls_per_minute)}), waiting {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    self.total_wait_time += wait_time
                    current_time = time.time()
            
            # Enforce minimum interval between calls
            if self.call_times:
                last_call = self.call_times[-1]
                time_since_last = current_time - last_call
                if time_since_last < self.min_interval:
                    sleep_time = self.min_interval - time_since_last
                    time.sleep(sleep_time)
                    self.total_wait_time += sleep_time
                    current_time = time.time()
            
            # Add current call
            self.call_times.append(current_time)
            self.total_calls += 1
    
    def hit_rate_limit(self, retry_after: float = None):
        """Called when we hit a rate limit."""
        with self.lock:
            self.is_throttled = True
            self.last_rate_limit_hit = time.time()
            self.consecutive_successes = 0
            self.total_rate_limit_hits += 1
            
            # Reduce rate by 50%
            old_rate = self.current_rate
            self.current_rate = max(int(self.current_rate * 0.5), 5)
            self.min_interval = 1.0 / self.current_rate
            
            print(f"  ⚠️  Rate limit hit #{self.total_rate_limit_hits}! Reducing from {old_rate} to {self.current_rate} calls/sec")
            
            # Clear recent calls to give breathing room
            self.call_times.clear()
            
            if retry_after:
                print(f"  ⏱️  Waiting {retry_after:.2f}s as requested by API...")
                time.sleep(retry_after)
                self.total_wait_time += retry_after
            else:
                # Default wait time
                default_wait = 2.0
                print(f"  ⏱️  Waiting {default_wait}s to recover...")
                time.sleep(default_wait)
                self.total_wait_time += default_wait
    
    def success(self):
        """Called after successful API call."""
        with self.lock:
            self.consecutive_successes += 1
            
            # Gradually increase rate after successes
            if self.is_throttled:
                # Need more successes when throttled
                threshold = 50 if self.current_rate < self.base_rate / 2 else 25
                
                if self.consecutive_successes >= threshold:
                    old_rate = self.current_rate
                    self.current_rate = min(int(self.current_rate * 1.5), self.base_rate)
                    self.min_interval = 1.0 / self.current_rate
                    self.consecutive_successes = 0  # Reset counter
                    
                    if self.current_rate >= self.base_rate:
                        self.is_throttled = False
                        print(f"  ✅ Rate limit fully recovered to {self.current_rate} calls/sec")
                    else:
                        print(f"  ↗️  Rate limit recovering: {old_rate} → {self.current_rate} calls/sec")
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        with self.lock:
            return {
                'total_calls': self.total_calls,
                'total_rate_limit_hits': self.total_rate_limit_hits,
                'total_wait_time': self.total_wait_time,
                'current_rate': self.current_rate,
                'base_rate': self.base_rate,
                'is_throttled': self.is_throttled,
                'calls_in_last_minute': len(self.call_times)
            }
    
    def reset(self):
        """Reset the rate limiter to initial state."""
        with self.lock:
            self.current_rate = self.base_rate
            self.min_interval = 1.0 / self.base_rate
            self.call_times.clear()
            self.consecutive_successes = 0
            self.is_throttled = False
            print(f"  🔄 Rate limiter reset to {self.base_rate} calls/sec")