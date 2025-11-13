import { Search, Bell, ChevronDown } from 'lucide-static';

export function Header() {
  const headerEl = document.createElement('header');
  headerEl.className = 'flex-shrink-0 bg-surface border-b border-border';
  
  headerEl.innerHTML = `
    <div class="flex items-center justify-between h-16 px-6 lg:px-8">
      <div class="flex items-center">
        <h1 id="page-title" class="text-xl font-semibold text-secondary-dark">Bảng điều khiển</h1>
      </div>
      <div class="flex items-center gap-4">
        <div class="relative">
          <span class="absolute left-3 top-1/2 -translate-y-1/2 text-secondary">${Search}</span>
          <input type="text" placeholder="Tìm kiếm..." class="w-64 bg-background border border-border rounded-lg pl-10 pr-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
        </div>
        <button class="relative text-secondary hover:text-primary">
          ${Bell}
          <span class="absolute top-0 right-0 w-2 h-2 bg-danger rounded-full"></span>
        </button>
        <div class="flex items-center gap-3">
          <img src="https://i.pravatar.cc/40?u=admin" alt="Admin" class="w-10 h-10 rounded-full">
          <div>
            <div class="font-semibold text-secondary-dark">Quản trị viên</div>
            <div class="text-sm text-secondary">admin@bank.com</div>
          </div>
          <button class="text-secondary">${ChevronDown}</button>
        </div>
      </div>
    </div>
  `;
  return headerEl;
}
