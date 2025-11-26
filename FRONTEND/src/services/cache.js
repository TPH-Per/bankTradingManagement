/**
 * Client-Side Caching Service
 * Provides in-memory caching for API responses to reduce server load and improve UX
 */

class CacheService {
  constructor() {
    this.cache = new Map();
    this.defaultTTL = 5 * 60 * 1000; // 5 minutes in milliseconds
    this.maxSize = 100; // Maximum number of cached items
  }

  /**
   * Generate cache key from endpoint and params
   */
  _generateKey(endpoint, params = {}) {
    const paramsStr = JSON.stringify(params);
    return `${endpoint}:${paramsStr}`;
  }

  /**
   * Get cached value
   */
  get(endpoint, params = {}) {
    const key = this._generateKey(endpoint, params);
    const item = this.cache.get(key);

    if (!item) {
      return null;
    }

    // Check if expired
    if (Date.now() > item.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return item.value;
  }

  /**
   * Set cached value
   */
  set(endpoint, params = {}, value, ttl = null) {
    const key = this._generateKey(endpoint, params);
    const expiresAt = Date.now() + (ttl || this.defaultTTL);

    // Enforce max size - remove oldest entry if needed
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, {
      value,
      expiresAt,
      cachedAt: Date.now(),
    });
  }

  /**
   * Invalidate cache by pattern
   */
  invalidate(pattern) {
    const regex = new RegExp(pattern);
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Clear all cache
   */
  clear() {
    this.cache.clear();
  }

  /**
   * Get cache stats
   */
  getStats() {
    const now = Date.now();
    let expired = 0;
    let active = 0;

    for (const item of this.cache.values()) {
      if (now > item.expiresAt) {
        expired++;
      } else {
        active++;
      }
    }

    return {
      total: this.cache.size,
      active,
      expired,
      maxSize: this.maxSize,
    };
  }

  /**
   * Clean expired entries
   */
  cleanup() {
    const now = Date.now();
    for (const [key, item] of this.cache.entries()) {
      if (now > item.expiresAt) {
        this.cache.delete(key);
      }
    }
  }
}

// Singleton instance
const cacheService = new CacheService();

// Auto-cleanup every 5 minutes
setInterval(() => {
  cacheService.cleanup();
}, 5 * 60 * 1000);

export default cacheService;

