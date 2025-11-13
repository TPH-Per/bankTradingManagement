import anime from 'animejs/lib/anime.es.js';
import { animatePageIn, animatePageOut } from './utils/animations.js';

const routes = {};
const contentEl = document.createElement('main');
contentEl.className = 'flex-1 p-6 lg:p-8 overflow-y-auto';

function navigate(path) {
  window.location.hash = path;
}

async function router() {
  const path = window.location.hash.slice(1) || '/';
  const routeHandler = routes[path] || routes['/404'];
  
  const renderView = async () => {
    contentEl.innerHTML = `<div class="w-full h-full flex items-center justify-center"><div class="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin"></div></div>`;
    try {
        const view = await routeHandler();
        contentEl.innerHTML = '';
        if (typeof view === 'string') {
            contentEl.innerHTML = view;
        } else {
            contentEl.appendChild(view);
        }
        animatePageIn(contentEl);
    } catch (error) {
        console.error("Error rendering route:", error);
        contentEl.innerHTML = `<div class="text-center text-danger"><h2>Error</h2><p>Could not load page.</p></div>`;
        animatePageIn(contentEl);
    }
  };

  if (routeHandler) {
    // If content is already present, animate it out before rendering the new view
    if (contentEl.hasChildNodes()) {
        animatePageOut(contentEl, renderView);
    } else {
        renderView();
    }
  }

  // Update active link in sidebar
  document.querySelectorAll('.nav-link').forEach(link => {
    const linkPath = new URL(link.href).hash.slice(1);
    if (linkPath === path) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}

export function addRoute(path, handler) {
  routes[path] = handler;
}

export function initRouter(appEl) {
  appEl.appendChild(contentEl);
  window.addEventListener('hashchange', router);
  window.addEventListener('load', () => {
      if (!window.location.hash) {
          window.location.hash = '/dashboard';
      } else {
          router();
      }
  });
}

export { navigate };
