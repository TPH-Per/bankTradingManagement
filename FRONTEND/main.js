import { initRouter, addRoute } from './src/router.js';
import { Sidebar } from './src/components/Sidebar.js';
import { Header } from './src/components/Header.js';
import { Dashboard } from './src/components/Dashboard.js';
import { Customers } from './src/components/Customers.js';
import { Transactions } from './src/components/Transactions.js';
import { Reports } from './src/components/Reports.js';
import { Settings } from './src/components/Settings.js';

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
  addRoute('/dashboard', Dashboard);
  addRoute('/customers', Customers);
  addRoute('/transactions', Transactions);
  addRoute('/reports', Reports);
  addRoute('/settings', Settings);
  addRoute('/404', () => '<h2>404 Not Found</h2>');
});
