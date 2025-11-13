import { Home, ArrowLeftRight, Users, BarChart, Settings, CircleDollarSign } from 'lucide-static';
import anime from 'animejs/lib/anime.es.js';

const navItems = [
  { path: '/dashboard', label: 'Bảng điều khiển', icon: Home },
  { path: '/transactions', label: 'Giao dịch', icon: ArrowLeftRight },
  { path: '/customers', label: 'Khách hàng', icon: Users },
  { path: '/reports', label: 'Báo cáo', icon: BarChart },
];

export function Sidebar() {
  const sidebarEl = document.createElement('aside');
  sidebarEl.className = 'w-64 bg-surface border-r border-border flex-shrink-0 flex flex-col p-4';

  sidebarEl.innerHTML = `
    <div id="logo-container" class="flex items-center gap-3 px-4 mb-8 cursor-pointer" title="BankFlow Home">
      <div id="app-logo" class="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
        <span class="text-background">
          ${CircleDollarSign}
        </span>
      </div>
      <span class="text-xl font-bold text-secondary-dark">BankFlow</span>
    </div>
    <nav class="flex-1 space-y-2">
      ${navItems.map(item => `
        <a href="#${item.path}" class="nav-link">
          <span class="icon">${item.icon}</span>
          <span>${item.label}</span>
        </a>
      `).join('')}
    </nav>
    <div class="mt-auto">
      <a href="#/settings" class="nav-link">
        <span class="icon">${Settings}</span>
        <span>Cài đặt</span>
      </a>
    </div>
  `;

  const logoContainer = sidebarEl.querySelector('#logo-container');
  if (logoContainer) {
    logoContainer.addEventListener('mouseenter', () => {
      anime({
        targets: '#app-logo',
        rotate: '360deg',
        duration: 600,
        easing: 'easeInOutSine',
      });
    });

    logoContainer.addEventListener('click', (e) => {
        e.preventDefault();
        window.location.hash = '/dashboard';
    });
  }

  return sidebarEl;
}
