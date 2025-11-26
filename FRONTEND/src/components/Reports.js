import { DollarSign, Users, CreditCard, Activity, LineChart, PieChart, BrainCircuit, TrendingUp, TrendingDown } from 'lucide-static';
import { Chart, registerables } from 'chart.js';
import { formatCurrency } from '../utils/helpers.js';
import { prepareFeatures, predictCashIn, predictCashOut, getReportsStats } from '../services/api.js';
import { animateStaggerIn } from '../utils/animations.js';

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

function initCharts(view, monthlyData, typeData) {
    if (volumeChartInstance) volumeChartInstance.destroy();
    if (typeChartInstance) typeChartInstance.destroy();

    const gridColor = 'hsla(220, 10%, 85%, 0.1)';
    const textColor = 'hsl(220, 10%, 65%)';

    const volumeCtx = view.querySelector('#volume-chart')?.getContext('2d');
    if (volumeCtx) {
        const labels = monthlyData?.labels || ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6', 'Tháng 7'];
        const values = monthlyData?.values || [0, 0, 0, 0, 0, 0, 0];

        volumeChartInstance = new Chart(volumeCtx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Khối lượng giao dịch (VND)',
                    data: values,
                    borderColor: 'hsl(160, 80%, 45%)',
                    backgroundColor: 'hsla(160, 80%, 45%, 0.1)',
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,
                plugins: {
                    legend: { labels: { color: textColor } },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return formatCurrency(context.parsed.y, 'VND');
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: textColor },
                        grid: { color: gridColor }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: textColor,
                            callback: function (value) {
                                return formatCurrency(value, 'VND');
                            }
                        },
                        grid: { color: gridColor }
                    }
                }
            }
        });
    }

    const typeCtx = view.querySelector('#type-chart')?.getContext('2d');
    if (typeCtx) {
        const typeLabels = [];
        const typeValues = [];
        const typeColors = ['hsl(160, 80%, 45%)', 'hsl(190, 80%, 60%)', 'hsl(145, 70%, 50%)', 'hsl(45, 90%, 60%)'];

        if (typeData) {
            if (typeData.P2P > 0) {
                typeLabels.push('Chuyển khoản P2P');
                typeValues.push(typeData.P2P);
            }
            if (typeData.cash_in > 0) {
                typeLabels.push('Nạp tiền');
                typeValues.push(typeData.cash_in);
            }
            if (typeData.cash_out > 0) {
                typeLabels.push('Rút tiền');
                typeValues.push(typeData.cash_out);
            }
            if (typeData.other > 0) {
                typeLabels.push('Khác');
                typeValues.push(typeData.other);
            }
        }

        // Fallback if no data
        if (typeLabels.length === 0) {
            typeLabels.push('Chưa có dữ liệu');
            typeValues.push(1);
        }

        typeChartInstance = new Chart(typeCtx, {
            type: 'doughnut',
            data: {
                labels: typeLabels,
                datasets: [{
                    label: 'Loại giao dịch',
                    data: typeValues,
                    backgroundColor: typeColors.slice(0, typeLabels.length),
                    borderColor: 'hsl(220, 20%, 16%)',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { color: textColor } },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                return `${label}: ${value.toLocaleString('vi-VN')} giao dịch`;
                            }
                        }
                    }
                }
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

    // Initial loading state
    view.innerHTML = `
        <div class="flex justify-between items-center mb-6">
            <h2 class="text-2xl font-bold text-secondary-dark">Báo cáo & Phân tích</h2>
            <div class="flex items-center gap-2">
                <label for="report-period" class="text-sm font-medium">Kỳ báo cáo:</label>
                <select id="report-period" class="bg-surface border border-border rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary">
                    <option value="30days">30 ngày qua</option>
                    <option value="90days">90 ngày qua</option>
                    <option value="year">Năm nay</option>
                </select>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div class="bg-surface p-6 rounded-xl shadow-md">
                <div class="skeleton-loader h-12 w-12 rounded-lg mb-4"></div>
                <div class="skeleton-loader h-8 w-32 mb-2"></div>
                <div class="skeleton-loader h-4 w-24"></div>
            </div>
            <div class="bg-surface p-6 rounded-xl shadow-md">
                <div class="skeleton-loader h-12 w-12 rounded-lg mb-4"></div>
                <div class="skeleton-loader h-8 w-32 mb-2"></div>
                <div class="skeleton-loader h-4 w-24"></div>
            </div>
            <div class="bg-surface p-6 rounded-xl shadow-md">
                <div class="skeleton-loader h-12 w-12 rounded-lg mb-4"></div>
                <div class="skeleton-loader h-8 w-32 mb-2"></div>
                <div class="skeleton-loader h-4 w-24"></div>
            </div>
            <div class="bg-surface p-6 rounded-xl shadow-md">
                <div class="skeleton-loader h-12 w-12 rounded-lg mb-4"></div>
                <div class="skeleton-loader h-8 w-32 mb-2"></div>
                <div class="skeleton-loader h-4 w-24"></div>
            </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-5 gap-6 mt-8">
            <div class="lg:col-span-3 bg-surface p-6 rounded-xl shadow-md">
                <h3 class="text-lg font-semibold text-secondary-dark mb-4 flex items-center gap-2">${LineChart} Khối lượng giao dịch hàng tháng</h3>
                <div class="h-80 skeleton-loader rounded-lg flex items-center justify-center text-secondary">Đang tải biểu đồ...</div>
            </div>
            <div class="lg:col-span-2 bg-surface p-6 rounded-xl shadow-md">
                <h3 class="text-lg font-semibold text-secondary-dark mb-4 flex items-center gap-2">${PieChart} Giao dịch theo loại</h3>
                <div class="h-80 skeleton-loader rounded-lg flex items-center justify-center text-secondary">Đang tải biểu đồ...</div>
            </div>
        </div>
        
        <div class="mt-8 bg-surface p-6 rounded-xl shadow-md" id="prediction-section">
            <h3 class="text-lg font-semibold text-secondary-dark mb-2 flex items-center gap-2">${BrainCircuit} Dự đoán M5P</h3>
            <p class="text-sm text-secondary mb-6">Dự đoán M5P cho ngày tiếp theo, tuần tiếp theo và tháng tiếp theo.</p>
            <div id="prediction-content"></div>
        </div>
    `;

    async function loadReportsData(period = '30days') {
        try {
            // Import getAllTransactions
            const { getAllTransactions } = await import('../services/api.js');

            // Fetch all transactions
            const response = await getAllTransactions();
            const allTransactions = response?.items || [];

            // Calculate date range based on period
            const now = new Date();
            let startDate = new Date();
            switch (period) {
                case '30days':
                    startDate.setDate(now.getDate() - 30);
                    break;
                case '90days':
                    startDate.setDate(now.getDate() - 90);
                    break;
                case 'year':
                    startDate.setFullYear(now.getFullYear(), 0, 1); // Jan 1st of current year
                    break;
            }

            // Filter transactions by date range
            const filteredTxs = allTransactions.filter(tx => {
                const txDate = new Date(tx.event_date || tx.event_ts);
                return txDate >= startDate && txDate <= now;
            });

            // Calculate stats
            let total_volume = 0;
            let transaction_count = filteredTxs.length;
            const activeCustomers = new Set();
            const monthlyVolumes = {};
            const transactionTypes = {
                P2P: 0,
                cash_in: 0,
                cash_out: 0,
                other: 0
            };

            filteredTxs.forEach(tx => {
                const amount = Math.abs(parseFloat(tx.amount || 0));
                total_volume += amount;

                // Track active customers
                if (tx.account_id) {
                    activeCustomers.add(tx.account_id);
                }

                // Group by month
                const txDate = new Date(tx.event_date || tx.event_ts);
                const monthKey = `${txDate.getFullYear()}-${String(txDate.getMonth() + 1).padStart(2, '0')}`;
                monthlyVolumes[monthKey] = (monthlyVolumes[monthKey] || 0) + amount;

                // Count by type
                const txType = (tx.transaction_type || '').toLowerCase();
                if (txType.includes('p2p') || txType.includes('transfer')) {
                    transactionTypes.P2P++;
                } else if (txType.includes('cash_in') || txType.includes('cashin') || tx.direction === 'CREDIT') {
                    transactionTypes.cash_in++;
                } else if (txType.includes('cash_out') || txType.includes('cashout') || tx.direction === 'DEBIT') {
                    transactionTypes.cash_out++;
                } else {
                    transactionTypes.other++;
                }
            });

            // Prepare monthly chart data (last 7 months)
            const monthLabels = [];
            const monthValues = [];
            for (let i = 6; i >= 0; i--) {
                const d = new Date();
                d.setMonth(d.getMonth() - i);
                const monthKey = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
                const monthName = d.toLocaleDateString('vi-VN', { month: 'short', year: 'numeric' });
                monthLabels.push(monthName);
                monthValues.push(monthlyVolumes[monthKey] || 0);
            }

            const stats = {
                total_volume,
                transaction_count,
                active_customers: activeCustomers.size,
                success_rate: 99.5, // Placeholder
                monthly_data: {
                    labels: monthLabels,
                    values: monthValues
                },
                transaction_types: transactionTypes
            };

            // Update stat cards
            const statsContainer = view.querySelector('.grid.grid-cols-1.md\\:grid-cols-2.lg\\:grid-cols-4');
            if (statsContainer) {
                statsContainer.innerHTML = `
                    ${StatCard({
                    icon: DollarSign,
                    title: 'Tổng khối lượng',
                    value: formatCurrency(stats.total_volume || 0, 'VND'),
                    change: 8.5,
                    delay: 0
                })}
                    ${StatCard({
                    icon: CreditCard,
                    title: 'Tổng giao dịch',
                    value: (stats.transaction_count || 0).toLocaleString('vi-VN'),
                    change: 12.3,
                    delay: 100
                })}
                    ${StatCard({
                    icon: Users,
                    title: 'Khách hàng hoạt động',
                    value: (stats.active_customers || 0).toLocaleString('vi-VN'),
                    change: 5.7,
                    delay: 200
                })}
                    ${StatCard({
                    icon: Activity,
                    title: 'Tỷ lệ thành công',
                    value: `${(stats.success_rate || 0).toFixed(1)}%`,
                    change: 0.2,
                    delay: 300
                })}
                `;
            }

            // Replace skeleton with canvas elements for charts
            const volumeChartContainer = view.querySelector('.lg\\:col-span-3 .skeleton-loader');
            if (volumeChartContainer) {
                volumeChartContainer.outerHTML = '<div class="relative" style="height: 320px;"><canvas id="volume-chart"></canvas></div>';
            }

            const typeChartContainer = view.querySelector('.lg\\:col-span-2 .skeleton-loader');
            if (typeChartContainer) {
                typeChartContainer.outerHTML = '<div class="relative" style="height: 320px;"><canvas id="type-chart"></canvas></div>';
            }

            // Update charts with real data
            setTimeout(() => {
                initCharts(view, stats.monthly_data, stats.transaction_types);
            }, 100);

            // Animate stat cards
            setTimeout(() => {
                animateStaggerIn(view.querySelectorAll('.bg-surface.p-6.rounded-xl.shadow-md'));
            }, 10);

        } catch (error) {
            console.error('Failed to load reports data:', error);
            const statsContainer = view.querySelector('.grid.grid-cols-1.md\\:grid-cols-2.lg\\:col-span-4');
            if (statsContainer) {
                statsContainer.innerHTML = `
                    <div class="col-span-4 bg-surface p-6 rounded-xl shadow-md text-center">
                        <p class="text-danger font-bold">Không thể tải dữ liệu báo cáo</p>
                        <p class="text-sm text-secondary mt-2">${error.message}</p>
                        <button onclick="location.reload()" class="mt-4 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark transition-colors">
                            Thử lại
                        </button>
                    </div>
                `;
            }
        }
    }

    // Load initial data
    loadReportsData('30days');

    // Handle period change
    const periodSelect = view.querySelector('#report-period');
    if (periodSelect) {
        periodSelect.addEventListener('change', (e) => {
            const period = e.target.value;
            loadReportsData(period);
        });
    }

    // Load predictions
    setTimeout(() => {
        loadPredictions(view);
    }, 10);

    return view;
}
