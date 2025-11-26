/**
 * Error Handling and Notification Service
 * Centralized error handling with user-friendly notifications
 */

class ErrorHandler {
  constructor() {
    this.notifications = [];
    this.maxNotifications = 5;
    this.listeners = new Set();
  }

  /**
   * Handle API error
   */
  handleError(error, context = {}) {
    console.error('Error:', error, context);

    let message = 'An unexpected error occurred';
    let type = 'error';
    let details = null;

    // Extract error message
    if (error.message) {
      message = error.message;
    } else if (typeof error === 'string') {
      message = error;
    }

    // Handle specific error types
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
      message = 'Connection failed. Please check your internet connection and ensure the server is running.';
      type = 'error';
    } else if (error.response) {
      const status = error.response.status;
      if (status === 401) {
        message = 'Authentication required. Please log in.';
        type = 'warning';
      } else if (status === 403) {
        message = 'You do not have permission to perform this action.';
        type = 'warning';
      } else if (status === 404) {
        message = 'The requested resource was not found.';
        type = 'warning';
      } else if (status === 429) {
        message = 'Too many requests. Please wait a moment and try again.';
        type = 'warning';
      } else if (status >= 500) {
        message = 'Server error. Please try again later.';
        type = 'error';
      } else {
        message = error.response.data?.detail || error.response.data?.message || message;
      }
      details = {
        status,
        statusText: error.response.statusText,
      };
    }

    // Create notification
    const notification = {
      id: Date.now() + Math.random(),
      message,
      type,
      details,
      context,
      timestamp: new Date().toISOString(),
    };

    this.addNotification(notification);
    return notification;
  }

  /**
   * Add notification
   */
  addNotification(notification) {
    this.notifications.unshift(notification);
    
    // Limit number of notifications
    if (this.notifications.length > this.maxNotifications) {
      this.notifications = this.notifications.slice(0, this.maxNotifications);
    }

    this.notifyListeners();
  }

  /**
   * Show success notification
   */
  showSuccess(message, details = null) {
    const notification = {
      id: Date.now() + Math.random(),
      message,
      type: 'success',
      details,
      timestamp: new Date().toISOString(),
    };
    this.addNotification(notification);
  }

  /**
   * Show warning notification
   */
  showWarning(message, details = null) {
    const notification = {
      id: Date.now() + Math.random(),
      message,
      type: 'warning',
      details,
      timestamp: new Date().toISOString(),
    };
    this.addNotification(notification);
  }

  /**
   * Show error notification
   */
  showError(message, details = null) {
    const notification = {
      id: Date.now() + Math.random(),
      message,
      type: 'error',
      details,
      timestamp: new Date().toISOString(),
    };
    this.addNotification(notification);
  }

  /**
   * Show info notification
   */
  showInfo(message, details = null) {
    const notification = {
      id: Date.now() + Math.random(),
      message,
      type: 'info',
      details,
      timestamp: new Date().toISOString(),
    };
    this.addNotification(notification);
  }

  /**
   * Remove notification
   */
  removeNotification(id) {
    this.notifications = this.notifications.filter(n => n.id !== id);
    this.notifyListeners();
  }

  /**
   * Clear all notifications
   */
  clear() {
    this.notifications = [];
    this.notifyListeners();
  }

  /**
   * Subscribe to notifications
   */
  subscribe(callback) {
    this.listeners.add(callback);
    callback(this.notifications);
    
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
        callback([...this.notifications]);
      } catch (error) {
        console.error('Error in notification listener:', error);
      }
    });
  }

  /**
   * Get all notifications
   */
  getNotifications() {
    return [...this.notifications];
  }
}

// Singleton instance
const errorHandler = new ErrorHandler();

export default errorHandler;

