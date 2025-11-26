/**
 * Notifications Component
 * Displays toast notifications for user feedback
 */

import errorHandler from '../services/errorHandler.js';

let notificationsContainer = null;
let unsubscribe = null;

function getNotificationClass(type) {
  const typeClasses = {
    success: 'bg-green-500 text-white',
    error: 'bg-red-500 text-white',
    warning: 'bg-yellow-500 text-white',
    info: 'bg-blue-500 text-white',
  };
  return typeClasses[type] || typeClasses.info;
}

function getIcon(type) {
  const icons = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ',
  };
  return icons[type] || icons.info;
}

function removeNotification(id) {
  errorHandler.removeNotification(id);
}

function renderNotifications(notifications) {
  if (!notificationsContainer) {
    notificationsContainer = document.createElement('div');
    notificationsContainer.className = 'fixed top-4 right-4 z-50 space-y-2 max-w-md';
    notificationsContainer.id = 'notifications-container';
    document.body.appendChild(notificationsContainer);
  }

  if (notifications.length === 0) {
    notificationsContainer.innerHTML = '';
    return;
  }

  notificationsContainer.innerHTML = notifications.map((notification, index) => `
    <div
      class="p-4 rounded-lg shadow-lg transform transition-all duration-300 ${getNotificationClass(notification.type)}"
      style="animation-delay: ${index * 0.1}s;"
    >
      <div class="flex items-start justify-between">
        <div class="flex items-start space-x-3 flex-1">
          <span class="text-xl font-bold">${getIcon(notification.type)}</span>
          <div class="flex-1">
            <p class="font-semibold">${notification.message}</p>
            ${notification.details && notification.details.status ? `
              <p class="text-sm mt-1 opacity-90">Status: ${notification.details.status}</p>
            ` : ''}
          </div>
        </div>
        <button
          onclick="window.removeNotification(${notification.id})"
          class="ml-4 text-white hover:text-gray-200 transition-colors cursor-pointer"
          aria-label="Close notification"
        >
          ✕
        </button>
      </div>
    </div>
  `).join('');

  // Attach remove handlers
  window.removeNotification = removeNotification;
}

export function initNotifications() {
  // Subscribe to notifications
  unsubscribe = errorHandler.subscribe((notifications) => {
    renderNotifications(notifications);
  });
}

export function destroyNotifications() {
  if (unsubscribe) {
    unsubscribe();
    unsubscribe = null;
  }
  if (notificationsContainer) {
    notificationsContainer.remove();
    notificationsContainer = null;
  }
}

