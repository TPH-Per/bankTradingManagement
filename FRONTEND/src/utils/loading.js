/**
 * Loading State Manager
 * Centralized loading state management for better UX
 */

class LoadingManager {
  constructor() {
    this.loadingStates = new Map();
    this.listeners = new Set();
  }

  /**
   * Set loading state for a key
   */
  setLoading(key, isLoading) {
    const previousState = this.loadingStates.get(key) || false;
    this.loadingStates.set(key, isLoading);
    
    // Notify listeners if state changed
    if (previousState !== isLoading) {
      this.notifyListeners();
    }
  }

  /**
   * Get loading state for a key
   */
  isLoading(key) {
    return this.loadingStates.get(key) || false;
  }

  /**
   * Check if any loading is in progress
   */
  isAnyLoading() {
    for (const isLoading of this.loadingStates.values()) {
      if (isLoading) return true;
    }
    return false;
  }

  /**
   * Get all loading states
   */
  getAllStates() {
    const states = {};
    for (const [key, isLoading] of this.loadingStates.entries()) {
      states[key] = isLoading;
    }
    return states;
  }

  /**
   * Clear loading state
   */
  clear(key) {
    this.loadingStates.delete(key);
    this.notifyListeners();
  }

  /**
   * Clear all loading states
   */
  clearAll() {
    this.loadingStates.clear();
    this.notifyListeners();
  }

  /**
   * Subscribe to loading state changes
   */
  subscribe(callback) {
    this.listeners.add(callback);
    callback(this.getAllStates());
    
    return () => {
      this.listeners.delete(callback);
    };
  }

  /**
   * Notify all listeners
   */
  notifyListeners() {
    const states = this.getAllStates();
    this.listeners.forEach(callback => {
      try {
        callback(states);
      } catch (error) {
        console.error('Error in loading state listener:', error);
      }
    });
  }
}

// Singleton instance
const loadingManager = new LoadingManager();

export default loadingManager;

