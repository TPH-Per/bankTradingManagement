import { initRouter, addRoute } from './src/router.js';
import { Sidebar } from './src/components/Sidebar.js';
import { Header } from './src/components/Header.js';
import { Dashboard } from './src/components/Dashboard.js';
import { Customers } from './src/components/Customers.js';
import { Transactions } from './src/components/Transactions.js';
import { Reports } from './src/components/Reports.js';
import { Settings } from './src/components/Settings.js';
import { HDFS } from './src/components/HDFS.js';
import { SystemStatus } from './src/components/SystemStatus.js';
import { initNetworkStatus } from './src/components/NetworkStatus.js';

document.addEventListener('DOMContentLoaded', () => {
  const appEl = document.getElementById('app');
  appEl.innerHTML = ''; // Clear any existing content

  // Main Layout
  const sidebar = Sidebar();
  const mainContentWrapper = document.createElement('div');
  mainContentWrapper.className = 'flex-1 flex flex-col overflow-hidden';
  const header = Header();

  mainContentWrapper.appendChild(header);

  appEl.appendChild(sidebar);
  appEl.appendChild(mainContentWrapper);

  // Initialize Router and add routes
  initRouter(mainContentWrapper);

  // Initialize production-ready features
  import('./src/components/Notifications.js').then((module) => {
    module.initNotifications();
  });

  import('./src/components/HealthStatus.js').then((module) => {
    module.initHealthStatus();
  });
  addRoute('/dashboard', Dashboard);
  addRoute('/customers', Customers);
  addRoute('/transactions', Transactions);
  addRoute('/reports', Reports);
  addRoute('/hdfs', HDFS);
  addRoute('/settings', Settings);
  addRoute('/system-status', SystemStatus);
  addRoute('/404', () => '<h2>404 Not Found</h2>');

  // Initialize network status indicator
  initNetworkStatus();
});
