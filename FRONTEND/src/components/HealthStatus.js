/**
 * Health Status Component
 * Displays backend service health status
 */

import healthService from '../services/health.js';

let healthContainer = null;
let unsubscribe = null;
let isExpanded = false;

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

function getServiceStatus(service) {
  if (!service) return 'unknown';
  return service.status || 'unknown';
}

function toggleExpanded() {
  isExpanded = !isExpanded;
  renderHealth();
}

function renderHealth(health) {
  if (!healthContainer) {
    healthContainer = document.createElement('div');
    healthContainer.className = 'fixed bottom-4 right-4 z-40 bg-gray-800 text-white rounded-lg shadow-lg overflow-hidden';
    healthContainer.id = 'health-status-container';
    document.body.appendChild(healthContainer);
  }

  if (!health) {
    healthContainer.innerHTML = `
      <div class="px-3 py-2 text-sm">
        <div class="flex items-center space-x-2">
          <div class="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
          <span>Checking health...</span>
        </div>
      </div>
    `;
    return;
  }

  const servicesHtml = health.services && Object.keys(health.services).length > 0 ? `
    <div>
      <div class="font-semibold mb-2">Services</div>
      <div class="space-y-1">
        ${Object.entries(health.services).map(([name, service]) => `
          <div class="flex justify-between items-center">
            <span class="opacity-75 capitalize">${name}:</span>
            <span class="px-2 py-0.5 rounded text-xs ${getStatusColor(getServiceStatus(service))}">
              ${getServiceStatus(service)}
            </span>
          </div>
        `).join('')}
      </div>
    </div>
  ` : '';

  const systemMetricsHtml = health.system ? `
    <div>
      <div class="font-semibold mb-2">System Metrics</div>
      <div class="space-y-1 text-xs">
        ${health.system.cpu_percent !== undefined ? `
          <div class="flex justify-between">
            <span class="opacity-75">CPU:</span>
            <span>${health.system.cpu_percent.toFixed(1)}%</span>
          </div>
        ` : ''}
        ${health.system.memory_percent !== undefined ? `
          <div class="flex justify-between">
            <span class="opacity-75">Memory:</span>
            <span>${health.system.memory_percent.toFixed(1)}%</span>
          </div>
        ` : ''}
        ${health.system.disk_percent !== undefined ? `
          <div class="flex justify-between">
            <span class="opacity-75">Disk:</span>
            <span>${health.system.disk_percent.toFixed(1)}%</span>
          </div>
        ` : ''}
      </div>
    </div>
  ` : '';

  const errorHtml = health.error ? `
    <div class="text-xs text-red-400 bg-red-900/20 p-2 rounded">
      <div class="font-semibold mb-1">Error:</div>
      <div>${health.error}</div>
    </div>
  ` : '';

  healthContainer.innerHTML = `
    <button
      onclick="window.toggleHealthExpanded()"
      class="w-full px-4 py-2 flex items-center justify-between hover:bg-gray-700 transition-colors cursor-pointer"
    >
      <div class="flex items-center space-x-2">
        <div class="w-2 h-2 rounded-full ${getStatusColor(health.status)}"></div>
        <span class="text-sm font-semibold">
          ${health.status === 'healthy' ? 'All Systems Operational' : 'Service Issues Detected'}
        </span>
      </div>
      <span class="text-xs opacity-75">${isExpanded ? '▼' : '▲'}</span>
    </button>
    ${isExpanded ? `
      <div class="px-4 py-3 border-t border-gray-700 max-h-96 overflow-y-auto">
        <div class="space-y-3 text-sm">
          <div>
            <div class="font-semibold mb-2">System Status</div>
            <div class="space-y-1">
              <div class="flex justify-between">
                <span class="opacity-75">Overall:</span>
                <span class="px-2 py-0.5 rounded text-xs ${getStatusColor(health.status)}">
                  ${health.status}
                </span>
              </div>
              ${health.lastCheck ? `
                <div class="flex justify-between text-xs opacity-60">
                  <span>Last Check:</span>
                  <span>${new Date(health.lastCheck).toLocaleTimeString()}</span>
                </div>
              ` : ''}
            </div>
          </div>
          ${servicesHtml}
          ${systemMetricsHtml}
          ${errorHtml}
        </div>
      </div>
    ` : ''}
  `;

  window.toggleHealthExpanded = toggleExpanded;
}

export function initHealthStatus() {
  // Start health monitoring
  healthService.startMonitoring();

  // Subscribe to health updates
  unsubscribe = healthService.subscribe((status) => {
    renderHealth(status);
  });
}

export function destroyHealthStatus() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
  healthService.stopMonitoring();
  if (healthContainer) {
    healthContainer.remove();
    healthContainer = null;
  }
}

