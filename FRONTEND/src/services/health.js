/**
 * Health Check Service
 * Monitors backend service health and status
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

class HealthService {
  constructor() {
    this.healthStatus = {
      status: 'unknown',
      lastCheck: null,
      services: {},
      system: {},
    };
    this.checkInterval = 30000; // 30 seconds
    this.intervalId = null;
    this.listeners = new Set();
  }

  /**
   * Check backend health
   */
  async checkHealth() {
    try {
      const response = await fetch(`${BASE_URL}/health/detailed`);
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      const data = await response.json();
      
      this.healthStatus = {
        ...data,
        lastCheck: new Date().toISOString(),
      };
      
      this.notifyListeners();
      return this.healthStatus;
    } catch (error) {
      this.healthStatus = {
        status: 'unhealthy',
        lastCheck: new Date().toISOString(),
        error: error.message,
        services: {},
      };
      this.notifyListeners();
      throw error;
    }
  }

  /**
   * Get simple health check
   */
  async getSimpleHealth() {
    try {
      const response = await fetch(`${BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      throw error;
    }
  }

  /**
   * Start automatic health monitoring
   */
  startMonitoring() {
    if (this.intervalId) {
      return; // Already monitoring
    }

    // Initial check
    this.checkHealth().catch(() => {
      // Ignore initial errors
    });

    // Periodic checks
    this.intervalId = setInterval(() => {
      this.checkHealth().catch(() => {
        // Ignore periodic errors
      });
    }, this.checkInterval);
  }

  /**
   * Stop automatic health monitoring
   */
  stopMonitoring() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Subscribe to health status changes
   */
  subscribe(callback) {
    this.listeners.add(callback);
    // Immediately call with current status
    callback(this.healthStatus);
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(callback);
    };
  }

  /**
   * Notify all listeners
   */
  notifyListeners() {
    this.listeners.forEach(callback => {
      try {
        callback(this.healthStatus);
      } catch (error) {
        console.error('Error in health status listener:', error);
      }
    });
  }

  /**
   * Get current health status
   */
  getStatus() {
    return { ...this.healthStatus };
  }

  /**
   * Check if service is healthy
   */
  isHealthy() {
    return this.healthStatus.status === 'healthy';
  }
}

// Singleton instance
const healthService = new HealthService();

export default healthService;

