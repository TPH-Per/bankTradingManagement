import { ArrowLeftRight, Calendar, Search, RefreshCw, TrendingUp, TrendingDown } from 'lucide-static';
import { formatCurrency, formatDate } from '../utils/helpers.js';
import { animateStaggerIn } from '../utils/animations.js';
import { getP2PAccountPairHistory, getP2PDirectionalHistory, getP2PCustomerPairHistory } from '../services/api.js';

function getCurrentYYYYMM() {
    const now = new Date();
    return parseInt(`${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}`);
}

function formatYYYYMM(month_yyyymm) {
    const year = Math.floor(month_yyyymm / 100);
    const month = month_yyyymm % 100;
    return `${year}-${String(month).padStart(2, '0')}`;
}

function TransactionRow(tx, index) {
    const isIncoming = tx.to_account || tx.to_customer_id;
    const directionClass = isIncoming ? 'text-success' : 'text-danger';
    const directionIcon = isIncoming ? TrendingUp : TrendingDown;

    return `
        <tr class="transaction-row border-b border-border hover:bg-secondary-light/50 transition-colors" style="animation-delay: ${index * 50}ms">
            <td class="p-4">
                <div class="flex items-center gap-2">
                    <div class="${directionClass}">${directionIcon}</div>
                    <div>
                        <div class="font-medium text-secondary-dark">${formatDate(tx.event_ts)}</div>
                        <div class="text-xs text-secondary">${tx.tx_id}</div>
                    </div>
                </div>
            </td>
            <td class="p-4">
                <div class="text-sm">
                    <div class="font-medium text-secondary-dark">Từ: ${tx.from_account || tx.from_customer_id || 'N/A'}</div>
                    <div class="text-secondary">Đến: ${tx.to_account || tx.to_customer_id || 'N/A'}</div>
                </div>
            </td>
            <td class="p-4 text-right">
                <div class="font-bold ${directionClass}">${formatCurrency(tx.amount, tx.currency || 'VND')}</div>
            </td>
            <td class="p-4">
                <span class="px-2 py-1 rounded text-xs ${tx.status === 'SETTLED' ? 'bg-green-900/50 text-success' : 'bg-yellow-900/50 text-warning'}">
                    ${tx.status || 'PENDING'}
                </span>
            </td>
            <td class="p-4">
                <div class="text-xs text-secondary max-w-xs truncate">
                    ${tx.extra_json?.description || 'P2P transfer'}
                </div>
            </td>
        </tr>
    `;
}

export function P2PHistory() {
    const view = document.createElement('div');
    view.className = 'p-6 space-y-6';

    let searchMode = 'account-pair'; // 'account-pair', 'directional', 'customer-pair'
    let transactions = [];

    view.innerHTML = `
        <div class="flex justify-between items-center">
            <div>
                <h2 class="text-2xl font-bold text-secondary-dark flex items-center gap-2">
                    ${ArrowLeftRight} Lịch sử P2P
                </h2>
                <p class="text-secondary mt-1">Xem lịch sử chuyển tiền giữa tài khoản / khách hàng</p>
            </div>
        </div>

        <!-- Search Mode Selector -->
        <div class="bg-surface rounded-xl shadow-md p-6">
            <div class="flex gap-4 mb-6">
                <label class="flex items-center gap-2 cursor-pointer">
                    <input type="radio" name="search-mode" value="account-pair" checked class="text-primary focus:ring-primary">
                    <span class="text-sm font-medium">Giữa 2 tài khoản (theo tháng)</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                    <input type="radio" name="search-mode" value="directional" class="text-primary focus:ring-primary">
                    <span class="text-sm font-medium">Từ A đến B (theo ngày)</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                    <input type="radio" name="search-mode" value="customer-pair" class="text-primary focus:ring-primary">
                    <span class="text-sm font-medium">Giữa 2 khách hàng (theo tháng)</span>
                </label>
            </div>

            <!-- Account Pair Search (default) -->
            <div id="account-pair-form" class="search-form">
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Tài khoản 1</label>
                        <input type="text" id="account1" placeholder="ACC_001" 
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Tài khoản 2</label>
                        <input type="text" id="account2" placeholder="ACC_002" 
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Tháng (YYYYMM)</label>
                        <input type="number" id="month-yyyymm" placeholder="202511" value="${getCurrentYYYYMM()}"
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div class="flex items-end">
                        <button id="search-account-pair-btn" class="w-full bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg flex items-center justify-center gap-2">
                            ${Search} Tìm kiếm
                        </button>
                    </div>
                </div>
            </div>

            <!-- Directional Search -->
            <div id="directional-form" class="search-form hidden">
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Từ tài khoản</label>
                        <input type="text" id="from-account" placeholder="ACC_001" 
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Đến tài khoản</label>
                        <input type="text" id="to-account" placeholder="ACC_002" 
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Ngày (YYYY-MM-DD)</label>
                        <input type="date" id="event-date" value="${new Date().toISOString().split('T')[0]}"
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div class="flex items-end">
                        <button id="search-directional-btn" class="w-full bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg flex items-center justify-center gap-2">
                            ${Search} Tìm kiếm
                        </button>
                    </div>
                </div>
            </div>

            <!-- Customer Pair Search -->
            <div id="customer-pair-form" class="search-form hidden">
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Khách hàng 1</label>
                        <input type="text" id="customer1" placeholder="CUST_001" 
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Khách hàng 2</label>
                        <input type="text" id="customer2" placeholder="CUST_002" 
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-secondary-dark mb-2">Tháng (YYYYMM)</label>
                        <input type="number" id="customer-month-yyyymm" placeholder="202511" value="${getCurrentYYYYMM()}"
                               class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
                    </div>
                    <div class="flex items-end">
                        <button id="search-customer-pair-btn" class="w-full bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg flex items-center justify-center gap-2">
                            ${Search} Tìm kiếm
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results -->
        <div class="bg-surface rounded-xl shadow-md overflow-hidden">
            <div class="p-6 border-b border-border">
                <div class="flex justify-between items-center">
                    <h3 class="text-lg font-semibold text-secondary-dark">Kết quả tìm kiếm</h3>
                    <div id="result-count" class="text-sm text-secondary"></div>
                </div>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead class="bg-secondary-light text-secondary-dark uppercase text-xs">
                        <tr>
                            <th class="p-4 text-left">Thời gian</th>
                            <th class="p-4 text-left">Chi tiết</th>
                            <th class="p-4 text-right">Số tiền</th>
                            <th class="p-4 text-left">Trạng thái</th>
                            <th class="p-4 text-left">Mô tả</th>
                        </tr>
                    </thead>
                    <tbody id="transactions-tbody">
                        <tr>
                            <td colspan="5" class="p-8 text-center text-secondary">
                                Nhập thông tin và nhấn "Tìm kiếm" để xem lịch sử giao dịch
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;

    const tbody = view.querySelector('#transactions-tbody');
    const resultCount = view.querySelector('#result-count');

    // Mode switching
    view.querySelectorAll('input[name="search-mode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            searchMode = e.target.value;
            view.querySelectorAll('.search-form').forEach(form => form.classList.add('hidden'));
            view.querySelector(`#${searchMode}-form`).classList.remove('hidden');

            // Clear results
            transactions = [];
            tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-secondary">Nhập thông tin và nhấn "Tìm kiếm"</td></tr>';
            resultCount.textContent = '';
        });
    });

    // Account pair search
    view.querySelector('#search-account-pair-btn').addEventListener('click', async () => {
        const account1 = view.querySelector('#account1').value.trim();
        const account2 = view.querySelector('#account2').value.trim();
        const monthYYYYMM = parseInt(view.querySelector('#month-yyyymm').value);

        if (!account1 || !account2 || !monthYYYYMM) {
            tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-danger">Vui lòng nhập đầy đủ thông tin</td></tr>';
            return;
        }

        tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-secondary">Đang tải...</td></tr>';

        try {
            const result = await getP2PAccountPairHistory({
                account_id1: account1,
                account_id2: account2,
                month_yyyymm: monthYYYYMM,
                limit: 100
            });

            transactions = result.items || [];
            if (transactions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-secondary">Không tìm thấy giao dịch nào</td></tr>';
                resultCount.textContent = '';
            } else {
                tbody.innerHTML = transactions.map((tx, idx) => TransactionRow(tx, idx)).join('');
                resultCount.textContent = `Tìm thấy ${transactions.length} giao dịch`;
                animateStaggerIn(view.querySelectorAll('.transaction-row'));
            }
        } catch (error) {
            tbody.innerHTML = `<tr><td colspan="5" class="p-8 text-center text-danger">Lỗi: ${error.message}</td></tr>`;
            resultCount.textContent = '';
        }
    });

    // Directional search
    view.querySelector('#search-directional-btn').addEventListener('click', async () => {
        const fromAccount = view.querySelector('#from-account').value.trim();
        const toAccount = view.querySelector('#to-account').value.trim();
        const eventDate = view.querySelector('#event-date').value;

        if (!fromAccount || !toAccount || !eventDate) {
            tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-danger">Vui lòng nhập đầy đủ thông tin</td></tr>';
            return;
        }

        tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-secondary">Đang tải...</td></tr>';

        try {
            const result = await getP2PDirectionalHistory({
                from_account: fromAccount,
                to_account: toAccount,
                event_date: eventDate,
                limit: 100
            });

            transactions = result.items || [];
            if (transactions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-secondary">Không tìm thấy giao dịch nào</td></tr>';
                resultCount.textContent = '';
            } else {
                tbody.innerHTML = transactions.map((tx, idx) => TransactionRow(tx, idx)).join('');
                resultCount.textContent = `Tìm thấy ${transactions.length} giao dịch`;
                animateStaggerIn(view.querySelectorAll('.transaction-row'));
            }
        } catch (error) {
            tbody.innerHTML = `<tr><td colspan="5" class="p-8 text-center text-danger">Lỗi: ${error.message}</td></tr>`;
            resultCount.textContent = '';
        }
    });

    // Customer pair search
    view.querySelector('#search-customer-pair-btn').addEventListener('click', async () => {
        const customer1 = view.querySelector('#customer1').value.trim();
        const customer2 = view.querySelector('#customer2').value.trim();
        const monthYYYYMM = parseInt(view.querySelector('#customer-month-yyyymm').value);

        if (!customer1 || !customer2 || !monthYYYYMM) {
            tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-danger">Vui lòng nhập đầy đủ thông tin</td></tr>';
            return;
        }

        tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-secondary">Đang tải...</td></tr>';

        try {
            const result = await getP2PCustomerPairHistory({
                customer_id1: customer1,
                customer_id2: customer2,
                month_yyyymm: monthYYYYMM,
                limit: 100
            });

            transactions = result.items || [];
            if (transactions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="p-8 text-center text-secondary">Không tìm thấy giao dịch nào</td></tr>';
                resultCount.textContent = '';
            } else {
                tbody.innerHTML = transactions.map((tx, idx) => TransactionRow(tx, idx)).join('');
                resultCount.textContent = `Tìm thấy ${transactions.length} giao dịch`;
                animateStaggerIn(view.querySelectorAll('.transaction-row'));
            }
        } catch (error) {
            tbody.innerHTML = `<tr><td colspan="5" class="p-8 text-center text-danger">Lỗi: ${error.message}</td></tr>`;
            resultCount.textContent = '';
        }
    });

    return view;
}
