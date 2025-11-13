import { DollarSign, Users, CreditCard, Activity, LineChart, PieChart, BrainCircuit, TrendingUp, TrendingDown } from 'lucide-static';
import { Chart, registerables } from 'chart.js';
import { formatCurrency } from '../utils/helpers.js';
import { prepareFeatures, predictCashIn, predictCashOut } from '../services/api.js';

Chart.register(...registerables);

let volumeChartInstance = null;
let typeChartInstance = null;

function StatCard({ icon, title, value, change, delay }) {
  const isPositive = change >= 0;
  return `
    <div class="bg-surface p-6 rounded-xl shadow-md transition-transform duration-300 hover:-translate-y-1" style="animation-delay: ${delay}ms">
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

function initCharts(view) {
    if (volumeChartInstance) volumeChartInstance.destroy();
    if (typeChartInstance) typeChartInstance.destroy();

    const gridColor = 'hsla(220, 10%, 85%, 0.1)';
    const textColor = 'hsl(220, 10%, 65%)';

    const volumeCtx = view.querySelector('#volume-chart')?.getContext('2d');
    if (volumeCtx) {
        volumeChartInstance = new Chart(volumeCtx, {
            type: 'line',
            data: {
                labels: ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6', 'Tháng 7'],
                datasets: [{
                    label: 'Khối lượng giao dịch', data: [65, 59, 80, 81, 56, 55, 40],
                    borderColor: 'hsl(160, 80%, 45%)', backgroundColor: 'hsla(160, 80%, 45%, 0.1)',
                    fill: true, tension: 0.4,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { color: textColor } } },
                scales: {
                    x: { ticks: { color: textColor }, grid: { color: gridColor } },
                    y: { ticks: { color: textColor }, grid: { color: gridColor } }
                }
            }
        });
    }

    const typeCtx = view.querySelector('#type-chart')?.getContext('2d');
    if (typeCtx) {
        typeChartInstance = new Chart(typeCtx, {
            type: 'doughnut',
            data: {
                labels: ['Chuyển khoản P2P', 'Thanh toán trực tuyến', 'Nạp tiền', 'Rút tiền'],
                datasets: [{
                    label: 'Loại giao dịch', data: [300, 150, 200, 50],
                    backgroundColor: ['hsl(160, 80%, 45%)', 'hsl(190, 80%, 60%)', 'hsl(145, 70%, 50%)', 'hsl(45, 90%, 60%)'],
                    borderColor: 'hsl(220, 20%, 16%)', borderWidth: 2,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { color: textColor } } }
            }
        });
    }
}

async function loadPredictions(view) {
    const predictionContent = view.querySelector('#prediction-content');
    if (!predictionContent) return;

    predictionContent.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div><div class="h-6 w-32 skeleton-loader mb-3"></div><div class="space-y-3"><div class="h-12 skeleton-loader"></div><div class="h-12 skeleton-loader"></div><div class="h-12 skeleton-loader"></div></div></div>
            <div><div class="h-6 w-32 skeleton-loader mb-3"></div><div class="space-y-3"><div class="h-12 skeleton-loader"></div><div class="h-12 skeleton-loader"></div><div class="h-12 skeleton-loader"></div></div></div>
        </div>`;

    try {
        const { features } = await prepareFeatures();
        const [cashInResult, cashOutResult] = await Promise.all([
            predictCashIn({ features }),
            predictCashOut({ features })
        ]);

        predictionContent.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                    <h4 class="text-md font-semibold text-success mb-4 flex items-center gap-2">${TrendingUp} Dự báo tiền vào</h4>
                    <div class="space-y-3">
                        <div class="flex justify-between items-center bg-secondary-light p-4 rounded-lg"><span class="text-secondary-dark font-medium">Ngày tiếp theo</span><span class="font-bold text-lg text-secondary-dark">${formatCurrency(cashInResult.next_day, 'VND')}</span></div>
                        <div class="flex justify-between items-center bg-secondary-light p-4 rounded-lg"><span class="text-secondary-dark font-medium">7 ngày tới</span><span class="font-bold text-lg text-secondary-dark">${formatCurrency(cashInResult.h7_sum, 'VND')}</span></div>
                        <div class="flex justify-between items-center bg-secondary-light p-4 rounded-lg"><span class="text-secondary-dark font-medium">Tháng tiếp theo</span><span class="font-bold text-lg text-secondary-dark">${formatCurrency(cashInResult.next_month_sum, 'VND')}</span></div>
                    </div>
                </div>
                <div>
                    <h4 class="text-md font-semibold text-danger mb-4 flex items-center gap-2">${TrendingDown} Dự báo tiền ra</h4>
                    <div class="space-y-3">
                        <div class="flex justify-between items-center bg-secondary-light p-4 rounded-lg"><span class="text-secondary-dark font-medium">Ngày tiếp theo</span><span class="font-bold text-lg text-secondary-dark">${formatCurrency(cashOutResult.next_day, 'VND')}</span></div>
                        <div class="flex justify-between items-center bg-secondary-light p-4 rounded-lg"><span class="text-secondary-dark font-medium">7 ngày tới</span><span class="font-bold text-lg text-secondary-dark">${formatCurrency(cashOutResult.h7_sum, 'VND')}</span></div>
                        <div class="flex justify-between items-center bg-secondary-light p-4 rounded-lg"><span class="text-secondary-dark font-medium">Tháng tiếp theo</span><span class="font-bold text-lg text-secondary-dark">${formatCurrency(cashOutResult.next_month_sum, 'VND')}</span></div>
                    </div>
                </div>
            </div>`;
    } catch (error) {
        predictionContent.innerHTML = `<div class="text-center p-8 text-danger"><p class="font-bold">Không thể tải dự đoán AI</p><p class="text-xs text-secondary mt-2 max-w-md mx-auto">${error.message}</p></div>`;
    }
}

export function Reports() {
    const view = document.createElement('div');
    view.innerHTML = `
        <div class="flex justify-between items-center mb-6">
            <h2 class="text-2xl font-bold text-secondary-dark">Báo cáo & Phân tích</h2>
            <div class="flex items-center gap-2">
                <label for="report-period" class="text-sm font-medium">Kỳ báo cáo:</label>
                <select id="report-period" class="bg-surface border border-border rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary">
                    <option>30 ngày qua</option><option>90 ngày qua</option><option>Năm nay</option>
                </select>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 animate-stagger-in">
            ${StatCard({ icon: DollarSign, title: 'Tổng khối lượng', value: formatCurrency(1250600210, 'VND'), change: 5.2, delay: 0 })}
            ${StatCard({ icon: CreditCard, title: 'Tổng giao dịch', value: '89,432', change: 2.1, delay: 100 })}
            ${StatCard({ icon: Users, title: 'Khách hàng hoạt động', value: '4,812', change: 1.5, delay: 200 })}
            ${StatCard({ icon: Activity, title: 'Tỷ lệ thành công', value: '99.2%', change: -0.2, delay: 300 })}
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-5 gap-6 mt-8">
            <div class="lg:col-span-3 bg-surface p-6 rounded-xl shadow-md animate-slide-in-up" style="animation-delay: 400ms">
                <h3 class="text-lg font-semibold text-secondary-dark mb-4 flex items-center gap-2">${LineChart} Khối lượng giao dịch hàng tháng</h3>
                <div class="h-80"><canvas id="volume-chart"></canvas></div>
            </div>
            <div class="lg:col-span-2 bg-surface p-6 rounded-xl shadow-md animate-slide-in-up" style="animation-delay: 500ms">
                <h3 class="text-lg font-semibold text-secondary-dark mb-4 flex items-center gap-2">${PieChart} Giao dịch theo loại</h3>
                <div class="h-80"><canvas id="type-chart"></canvas></div>
            </div>
        </div>
        
        <div class="mt-8 bg-surface p-6 rounded-xl shadow-md animate-slide-in-up" id="prediction-section" style="animation-delay: 600ms">
            <h3 class="text-lg font-semibold text-secondary-dark mb-2 flex items-center gap-2">${BrainCircuit} Dự đoán M5P</h3>
            <p class="text-sm text-secondary mb-6">Dự đoán M5P cho ngày tiếp theo, tuần tiếp theo và tháng tiếp theo.</p>
            <div id="prediction-content"></div>
        </div>
    `;

    setTimeout(() => {
        initCharts(view);
        loadPredictions(view);
    }, 10);

    return view;
}
