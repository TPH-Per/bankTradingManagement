/**
 * Performance Widget
 * Mini performance monitor for dashboard
 */

import cacheService from '../services/cache.js';

let performanceData = {
  apiCalls: 0,
  cacheHits: 0,
  cacheMisses: 0,
  avgResponseTime: 0,
  responseTimes: [],
};

export function PerformanceWidget() {
  const widget = document.createElement('div');
  widget.className = 'bg-surface p-4 rounded-xl shadow-md';
  
  function updateWidget() {
    const cacheStats = cacheService.getStats();
    const hitRate = performanceData.cacheHits + performanceData.cacheMisses > 0
      ? ((performanceData.cacheHits / (performanceData.cacheHits + performanceData.cacheMisses)) * 100).toFixed(1)
      : 0;

    widget.innerHTML = `
      <h3 class="text-lg font-semibold mb-3 text-secondary-dark">Performance</h3>
      <div class="space-y-3">
        <div class="flex justify-between items-center">
          <span class="text-sm text-secondary">Cache Hit Rate</span>
          <span class="font-semibold text-primary">${hitRate}%</span>
        </div>
        <div class="w-full bg-background rounded-full h-2">
          <div class="bg-primary h-2 rounded-full transition-all" style="width: ${hitRate}%"></div>
        </div>
        <div class="flex justify-between items-center text-sm">
          <span class="text-secondary">API Calls</span>
          <span class="font-semibold">${performanceData.apiCalls}</span>
        </div>
        ${performanceData.avgResponseTime > 0 ? `
          <div class="flex justify-between items-center text-sm">
            <span class="text-secondary">Avg Response</span>
            <span class="font-semibold">${performanceData.avgResponseTime.toFixed(0)}ms</span>
          </div>
        ` : ''}
        <div class="flex justify-between items-center text-sm">
          <span class="text-secondary">Cache Entries</span>
          <span class="font-semibold">${cacheStats?.active || 0}</span>
        </div>
      </div>
    `;
  }

  // Track API calls
  const originalFetch = window.fetch;
  window.fetch = async function(...args) {
    const startTime = performance.now();
    performanceData.apiCalls++;
    
    try {
      const response = await originalFetch(...args);
      const endTime = performance.now();
      const responseTime = endTime - startTime;
      
      performanceData.responseTimes.push(responseTime);
      if (performanceData.responseTimes.length > 50) {
        performanceData.responseTimes.shift();
      }
      performanceData.avgResponseTime = performanceData.responseTimes.reduce((a, b) => a + b, 0) / performanceData.responseTimes.length;
      
      // Check if response was from cache (this is approximate)
      if (args[0] && typeof args[0] === 'string' && args[0].includes('/api/')) {
        const cached = cacheService.get(args[0], {});
        if (cached) {
          performanceData.cacheHits++;
        } else {
          performanceData.cacheMisses++;
        }
      }
      
      updateWidget();
      return response;
    } catch (error) {
      updateWidget();
      throw error;
    }
  };

  // Update periodically
  const interval = setInterval(updateWidget, 2000);
  
  // Initial render
  updateWidget();

  // Cleanup
  widget.addEventListener('destroy', () => {
    clearInterval(interval);
    window.fetch = originalFetch;
  });

  return widget;
}

