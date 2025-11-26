/**
 * Request Retry Service
 * Implements exponential backoff retry logic for failed requests
 */

class RetryService {
  constructor() {
    this.maxRetries = 3;
    this.baseDelay = 1000; // 1 second
    this.maxDelay = 10000; // 10 seconds
    this.retryableStatuses = [408, 429, 500, 502, 503, 504];
  }

  /**
   * Check if error is retryable
   */
  isRetryable(error, status) {
    // Network errors are always retryable
    if (error && (error.name === 'TypeError' || error.message.includes('Failed to fetch'))) {
      return true;
    }

    // HTTP status codes that are retryable
    return status && this.retryableStatuses.includes(status);
  }

  /**
   * Calculate delay with exponential backoff
   */
  calculateDelay(attempt) {
    const delay = Math.min(
      this.baseDelay * Math.pow(2, attempt),
      this.maxDelay
    );
    // Add jitter to prevent thundering herd
    const jitter = Math.random() * 0.3 * delay;
    return delay + jitter;
  }

  /**
   * Retry a function with exponential backoff
   */
  async retry(fn, options = {}) {
    const maxRetries = options.maxRetries || this.maxRetries;
    let lastError;
    let lastStatus;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const response = await fn();
        return response;
      } catch (error) {
        lastError = error;
        
        // Try to extract status from error
        if (error.response) {
          lastStatus = error.response.status;
        } else if (error.status) {
          lastStatus = error.status;
        }

        // Don't retry on last attempt
        if (attempt === maxRetries) {
          break;
        }

        // Check if error is retryable
        if (!this.isRetryable(error, lastStatus)) {
          throw error;
        }

        // Wait before retrying
        const delay = this.calculateDelay(attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    // All retries exhausted
    throw lastError;
  }
}

// Singleton instance
const retryService = new RetryService();

export default retryService;

