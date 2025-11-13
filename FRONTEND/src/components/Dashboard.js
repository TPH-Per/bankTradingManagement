import { DollarSign, Users, Activity, CreditCard } from 'lucide-static';
import { formatCurrency, formatDate } from '../utils/helpers.js';
import { animateStaggerIn } from '../utils/animations.js';

function StatCard({ icon, title, value, change }) {
  const isPositive = change >= 0;
  return `
    <div class="stat-card bg-surface p-6 rounded-xl shadow-md transition-transform duration-300 hover:-translate-y-1">
      <div class="flex justify-between items-start">
        <div class="flex items-center justify-center w-12 h-12 rounded-lg bg-secondary-light text-primary">
          ${icon}
        </div>
        <div class="flex items-center gap-1 text-sm font-medium ${isPositive ? 'text-success' : 'text-danger'}">
          <span>${isPositive ? '▲' : '▼'}</span>
          <span>${Math.abs(change)}%</span>
        </div>
      </div>
      <div class="mt-4">
        <p class="text-3xl font-bold text-secondary-dark">${value}</p>
        <p class="text-sm text-secondary">${title}</p>
      </div>
    </div>
  `;
}

function RecentTransactionItem({ name, date, amount, currency, status }) {
    const isPositive = amount >= 0;
    const statusClasses = status === 'Hoàn thành' ? 'bg-green-900/50 text-success' : 'bg-yellow-900/50 text-warning';
    return `
        <div class="transaction-item flex items-center justify-between py-3">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-full bg-secondary-light flex items-center justify-center font-bold text-primary">${name.charAt(0)}</div>
                <div>
                    <p class="font-semibold text-secondary-dark">${name}</p>
                    <p class="text-sm text-secondary">${formatDate(date)}</p>
                </div>
            </div>
            <div class="text-right">
                <p class="font-semibold ${isPositive ? 'text-success' : 'text-danger'}">${formatCurrency(amount, currency)}</p>
                <p class="text-xs px-2 py-0.5 rounded-full inline-block ${statusClasses}">${status}</p>
            </div>
        </div>
    `;
}

export function Dashboard() {
    const view = document.createElement('div');
    view.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            ${StatCard({ icon: DollarSign, title: 'Tổng doanh thu', value: formatCurrency(405300750, 'VND'), change: 12.5 })}
            ${StatCard({ icon: Users, title: 'Khách hàng mới', value: '1,204', change: 8.2 })}
            ${StatCard({ icon: CreditCard, title: 'Giao dịch', value: '23,890', change: -1.8 })}
            ${StatCard({ icon: Activity, title: 'Tình trạng hệ thống', value: '99.8%', change: 0.1 })}
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
            <div class="lg:col-span-2 bg-surface p-6 rounded-xl shadow-md">
                <h3 class="text-lg font-semibold text-secondary-dark mb-4">Tổng quan doanh thu</h3>
                <div class="h-80 skeleton-loader rounded-lg flex items-center justify-center text-secondary">
                    <!-- Biểu đồ sẽ được hiển thị ở đây -->
                </div>
            </div>
            <div class="bg-surface p-6 rounded-xl shadow-md">
                <h3 class="text-lg font-semibold text-secondary-dark mb-4">Giao dịch gần đây</h3>
                <div class="space-y-2">
                    ${RecentTransactionItem({ name: 'John Doe', date: new Date(), amount: 2500000, currency: 'VND', status: 'Hoàn thành' })}
                    ${RecentTransactionItem({ name: 'Jane Smith', date: new Date(), amount: -150500, currency: 'VND', status: 'Hoàn thành' })}
                    ${RecentTransactionItem({ name: 'Online Store', date: new Date(), amount: -59990, currency: 'VND', status: 'Đang xử lý' })}
                    ${RecentTransactionItem({ name: 'Michael Brown', date: new Date(), amount: 5000000, currency: 'VND', status: 'Hoàn thành' })}
                </div>
            </div>
        </div>
    `;

    // Animate elements on load
    setTimeout(() => {
        animateStaggerIn(view.querySelectorAll('.stat-card, .lg\\:col-span-2, .bg-surface:not(.stat-card)'));
        animateStaggerIn(view.querySelectorAll('.transaction-item'));
    }, 10);

    return view;
}
