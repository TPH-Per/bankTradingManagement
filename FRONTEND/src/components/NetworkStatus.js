/**
 * Network Status Indicator
 * Shows connection status, retry attempts, and network health
 */

import healthService from '../services/health.js';

let statusIndicator = null;
let unsubscribe = null;
let isOnline = navigator.onLine;
let retryCount = 0;

export function initNetworkStatus() {
  // Create status indicator
  statusIndicator = document.createElement('div');
  statusIndicator.id = 'network-status';
  statusIndicator.className = 'fixed top-16 right-4 z-30 bg-gray-800 text-white px-3 py-2 rounded-lg shadow-lg text-sm';
  
  // Listen to online/offline events
  window.addEventListener('online', () => {
    isOnline = true;
    updateStatus();
  });
  
  window.addEventListener('offline', () => {
    isOnline = false;
    updateStatus();
  });

  // Subscribe to health service
  unsubscribe = healthService.subscribe((health) => {
    updateStatus(health);
  });

  // Initial update
  updateStatus();
  
  // Add to body
  document.body.appendChild(statusIndicator);
}

function updateStatus(health = null) {
  if (!statusIndicator) return;

  const healthStatus = health || healthService.getStatus();
  const isHealthy = healthStatus?.status === 'healthy';
  
  if (!isOnline) {
    statusIndicator.innerHTML = `
      <div class="flex items-center space-x-2">
        <div class="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
        <span>Offline</span>
      </div>
    `;
    statusIndicator.className = 'fixed top-16 right-4 z-30 bg-red-600 text-white px-3 py-2 rounded-lg shadow-lg text-sm';
    return;
  }

  if (!isHealthy) {
    statusIndicator.innerHTML = `
      <div class="flex items-center space-x-2">
        <div class="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
        <span>Service Issues</span>
      </div>
    `;
    statusIndicator.className = 'fixed top-16 right-4 z-30 bg-yellow-600 text-white px-3 py-2 rounded-lg shadow-lg text-sm';
    return;
  }

  statusIndicator.innerHTML = `
    <div class="flex items-center space-x-2">
      <div class="w-2 h-2 bg-green-500 rounded-full"></div>
      <span>Online</span>
    </div>
  `;
  statusIndicator.className = 'fixed top-16 right-4 z-30 bg-gray-800 text-white px-3 py-2 rounded-lg shadow-lg text-sm';
}

export function destroyNetworkStatus() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
  if (statusIndicator) {
    statusIndicator.remove();
    statusIndicator = null;
  }
  window.removeEventListener('online', updateStatus);
  window.removeEventListener('offline', updateStatus);
}

