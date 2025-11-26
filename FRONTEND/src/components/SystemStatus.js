/**
 * System Status Dashboard
 * Comprehensive view of system health, metrics, and performance
 */

import healthService from '../services/health.js';
import cacheService from '../services/cache.js';
import errorHandler from '../services/errorHandler.js';
import loadingManager from '../utils/loading.js';

let healthUnsubscribe = null;
let cacheStatsInterval = null;
let performanceMetrics = {
  apiCalls: 0,
  cacheHits: 0,
  cacheMisses: 0,
  retries: 0,
  errors: 0,
  avgResponseTime: 0,
  responseTimes: [],
};

export function SystemStatus() {
  const view = document.createElement('div');
  view.className = 'p-6 space-y-6';
  
  let healthStatus = null;
  let cacheStats = null;
  let loadingStates = {};

  function updateView() {
    view.innerHTML = `
      <div class="mb-6">
        <h1 class="text-3xl font-bold text-secondary-dark mb-2">System Status</h1>
        <p class="text-secondary">Real-time monitoring of system health and performance</p>
      </div>

      <!-- Overall Status Card -->
      <div class="bg-surface rounded-xl shadow-md p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4 text-secondary-dark">Overall System Status</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          ${renderStatusCard('System Health', healthStatus?.status || 'unknown', getStatusColor(healthStatus?.status))}
          ${renderStatusCard('Cache Status', cacheStats ? `${cacheStats.active} active entries` : 'N/A', 'bg-blue-500')}
          ${renderStatusCard('Active Operations', Object.values(loadingStates).filter(Boolean).length, 'bg-purple-500')}
        </div>
      </div>

      <!-- Services Health -->
      <div class="bg-surface rounded-xl shadow-md p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4 text-secondary-dark">Services Health</h2>
        <div class="space-y-3">
          ${healthStatus?.services ? Object.entries(healthStatus.services).map(([name, service]) => `
            <div class="flex items-center justify-between p-3 bg-background rounded-lg">
              <div class="flex items-center space-x-3">
                <div class="w-3 h-3 rounded-full ${getStatusColor(service.status || 'unknown')}"></div>
                <span class="font-medium capitalize">${name}</span>
              </div>
              <div class="flex items-center space-x-4">
                <span class="text-sm text-secondary">${service.status || 'unknown'}</span>
                ${service.error ? `<span class="text-xs text-red-500">${service.error}</span>` : ''}
              </div>
            </div>
          `).join('') : '<p class="text-secondary">No service data available</p>'}
        </div>
      </div>

      <!-- System Metrics -->
      ${healthStatus?.system ? `
        <div class="bg-surface rounded-xl shadow-md p-6 mb-6">
          <h2 class="text-xl font-semibold mb-4 text-secondary-dark">System Metrics</h2>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            ${renderMetricCard('CPU Usage', `${healthStatus.system.cpu_percent?.toFixed(1) || 0}%`, healthStatus.system.cpu_percent, 'CPU')}
            ${renderMetricCard('Memory Usage', `${healthStatus.system.memory_percent?.toFixed(1) || 0}%`, healthStatus.system.memory_percent, 'Memory')}
            ${renderMetricCard('Disk Usage', `${healthStatus.system.disk_percent?.toFixed(1) || 0}%`, healthStatus.system.disk_percent, 'Disk')}
          </div>
        </div>
      ` : ''}

      <!-- Performance Metrics -->
      <div class="bg-surface rounded-xl shadow-md p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4 text-secondary-dark">Performance Metrics</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          ${renderMetricCard('API Calls', performanceMetrics.apiCalls, null, 'Total')}
          ${renderMetricCard('Cache Hit Rate', cacheStats ? `${((cacheStats.active / (cacheStats.active + performanceMetrics.cacheMisses)) * 100).toFixed(1)}%` : '0%', null, 'Hit Rate')}
          ${renderMetricCard('Retries', performanceMetrics.retries, null, 'Total')}
          ${renderMetricCard('Errors', performanceMetrics.errors, null, 'Total')}
        </div>
        ${performanceMetrics.avgResponseTime > 0 ? `
          <div class="mt-4">
            <div class="flex justify-between text-sm mb-2">
              <span class="text-secondary">Average Response Time</span>
              <span class="font-semibold">${performanceMetrics.avgResponseTime.toFixed(2)}ms</span>
            </div>
            <div class="w-full bg-background rounded-full h-2">
              <div class="bg-primary h-2 rounded-full" style="width: ${Math.min((performanceMetrics.avgResponseTime / 1000) * 100, 100)}%"></div>
            </div>
          </div>
        ` : ''}
      </div>

      <!-- Cache Statistics -->
      <div class="bg-surface rounded-xl shadow-md p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4 text-secondary-dark">Cache Statistics</h2>
        ${cacheStats ? `
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="text-center p-4 bg-background rounded-lg">
              <div class="text-2xl font-bold text-primary">${cacheStats.total}</div>
              <div class="text-sm text-secondary mt-1">Total Entries</div>
            </div>
            <div class="text-center p-4 bg-background rounded-lg">
              <div class="text-2xl font-bold text-green-500">${cacheStats.active}</div>
              <div class="text-sm text-secondary mt-1">Active</div>
            </div>
            <div class="text-center p-4 bg-background rounded-lg">
              <div class="text-2xl font-bold text-yellow-500">${cacheStats.expired || 0}</div>
              <div class="text-sm text-secondary mt-1">Expired</div>
            </div>
            <div class="text-center p-4 bg-background rounded-lg">
              <div class="text-2xl font-bold text-blue-500">${cacheStats.maxSize}</div>
              <div class="text-sm text-secondary mt-1">Max Size</div>
            </div>
          </div>
          <div class="mt-4 flex space-x-2">
            <button onclick="window.clearCache()" class="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors">
              Clear Cache
            </button>
            <button onclick="window.cleanupCache()" class="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors">
              Cleanup Expired
            </button>
          </div>
        ` : '<p class="text-secondary">No cache statistics available</p>'}
      </div>

      <!-- Active Loading States -->
      <div class="bg-surface rounded-xl shadow-md p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4 text-secondary-dark">Active Operations</h2>
        ${Object.keys(loadingStates).length > 0 ? `
          <div class="space-y-2">
            ${Object.entries(loadingStates).filter(([_, isLoading]) => isLoading).map(([key, _]) => `
              <div class="flex items-center justify-between p-3 bg-background rounded-lg">
                <span class="text-sm font-medium">${key}</span>
                <div class="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
              </div>
            `).join('')}
          </div>
        ` : '<p class="text-secondary">No active operations</p>'}
      </div>

      <!-- Recent Errors -->
      <div class="bg-surface rounded-xl shadow-md p-6">
        <h2 class="text-xl font-semibold mb-4 text-secondary-dark">Recent Errors</h2>
        ${errorHandler.getNotifications().filter(n => n.type === 'error').slice(0, 5).length > 0 ? `
          <div class="space-y-2">
            ${errorHandler.getNotifications().filter(n => n.type === 'error').slice(0, 5).map(notification => `
              <div class="p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                <div class="flex justify-between items-start">
                  <div class="flex-1">
                    <p class="text-sm font-medium text-red-400">${notification.message}</p>
                    ${notification.context?.endpoint ? `
                      <p class="text-xs text-secondary mt-1">${notification.context.method || 'GET'} ${notification.context.endpoint}</p>
                    ` : ''}
                    <p class="text-xs text-secondary mt-1">${new Date(notification.timestamp).toLocaleString()}</p>
                  </div>
                </div>
              </div>
            `).join('')}
          </div>
        ` : '<p class="text-secondary">No recent errors</p>'}
      </div>
    `;

    // Attach event handlers
    window.clearCache = () => {
      cacheService.clear();
      updateView();
      errorHandler.showSuccess('Cache cleared successfully');
    };

    window.cleanupCache = () => {
      cacheService.cleanup();
      updateView();
      errorHandler.showSuccess('Expired cache entries cleaned up');
    };
  }

  function renderStatusCard(title, value, colorClass) {
    return `
      <div class="p-4 bg-background rounded-lg">
        <div class="flex items-center space-x-3 mb-2">
          <div class="w-3 h-3 rounded-full ${colorClass || 'bg-gray-500'}"></div>
          <span class="text-sm text-secondary">${title}</span>
        </div>
        <div class="text-2xl font-bold text-secondary-dark">${value}</div>
      </div>
    `;
  }

  function renderMetricCard(title, value, percent, unit) {
    const barWidth = percent ? Math.min(percent, 100) : 0;
    const barColor = percent ? (percent > 80 ? 'bg-red-500' : percent > 60 ? 'bg-yellow-500' : 'bg-green-500') : 'bg-primary';
    
    return `
      <div class="p-4 bg-background rounded-lg">
        <div class="flex justify-between items-start mb-2">
          <span class="text-sm text-secondary">${title}</span>
          <span class="text-lg font-bold text-secondary-dark">${value}</span>
        </div>
        ${percent !== null ? `
          <div class="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div class="${barColor} h-2 rounded-full transition-all" style="width: ${barWidth}%"></div>
          </div>
        ` : ''}
      </div>
    `;
  }

  function getStatusColor(status) {
    switch (status) {
      case 'healthy':
        return 'bg-green-500';
      case 'degraded':
        return 'bg-yellow-500';
      case 'unhealthy':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  }

  // Subscribe to health updates
  healthUnsubscribe = healthService.subscribe((status) => {
    healthStatus = status;
    updateView();
  });

  // Update cache stats periodically
  cacheStatsInterval = setInterval(() => {
    cacheStats = cacheService.getStats();
    updateView();
  }, 2000);

  // Subscribe to loading states
  loadingManager.subscribe((states) => {
    loadingStates = states;
    updateView();
  });

  // Track API performance (intercept fetch)
  const originalFetch = window.fetch;
  window.fetch = async function(...args) {
    const startTime = performance.now();
    performanceMetrics.apiCalls++;
    
    try {
      const response = await originalFetch(...args);
      const endTime = performance.now();
      const responseTime = endTime - startTime;
      
      performanceMetrics.responseTimes.push(responseTime);
      if (performanceMetrics.responseTimes.length > 100) {
        performanceMetrics.responseTimes.shift();
      }
      performanceMetrics.avgResponseTime = performanceMetrics.responseTimes.reduce((a, b) => a + b, 0) / performanceMetrics.responseTimes.length;
      
      if (!response.ok) {
        performanceMetrics.errors++;
      }
      
      return response;
    } catch (error) {
      performanceMetrics.errors++;
      throw error;
    }
  };

  // Initial render
  healthStatus = healthService.getStatus();
  cacheStats = cacheService.getStats();
  loadingStates = loadingManager.getAllStates();
  updateView();

  // Cleanup on destroy
  view.addEventListener('destroy', () => {
    if (healthUnsubscribe) healthUnsubscribe();
    if (cacheStatsInterval) clearInterval(cacheStatsInterval);
    window.fetch = originalFetch;
  });

  return view;
}

