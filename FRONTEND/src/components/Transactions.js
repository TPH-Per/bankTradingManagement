import { ArrowRight, Landmark, Download } from 'lucide-static';
import { formatCurrency, formatDate } from '../utils/helpers.js';
import { createTransaction, getAllTransactions } from '../services/api.js';
import { animateStaggerIn, showResponseMessage } from '../utils/animations.js';

function TransactionLogRow(tx) {
    const isPositive = tx.amount >= 0;
    const statusMap = {
        'Completed': 'bg-green-900/50 text-success',
        'Pending': 'bg-yellow-900/50 text-warning',
        'Failed': 'bg-red-900/50 text-danger',
    };
    const statusClass = statusMap[tx.status] || 'bg-gray-700 text-gray-400';

    let transactionLabel = '';
    if (tx.type === 'p2p') {
        transactionLabel = '<span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded-full bg-blue-900/50 text-blue-300">Giao dịch khách hàng</span>';
    } else if (tx.type === 'cash_in' || tx.type === 'cash_out') {
        transactionLabel = '<span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded-full bg-purple-900/50 text-purple-300">Giao dịch ngân hàng</span>';
    } else {
        transactionLabel = `<span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded-full bg-gray-700 text-gray-400">Giao dịch không xác định (${tx.type})</span>`;
    }

    return `
        <tr class="tx-log-row border-b border-border last:border-0">
            <td class="p-3">${tx.transaction_id || 'N/A'}</td>
            <td class="p-3">${formatDate(tx.created_at)}</td>
            <td class="p-3">
                <div class="font-medium text-secondary-dark">${tx.description || 'N/A'} ${transactionLabel}</div>
                <div class="text-xs text-secondary">${tx.type}</div>
            </td>
            <td class="p-3 font-mono text-right ${isPositive ? 'text-success' : 'text-danger'}">
                ${formatCurrency(tx.amount, 'VND')}
            </td>
            <td class="p-3 text-center">
                <span class="px-2 py-1 text-xs font-semibold rounded-full ${statusClass}">
                    ${tx.status}
                </span>
            </td>
        </tr>
    `;
}

function SkeletonLogRow() {
    return `
        <tr class="border-b border-border last:border-0">
            <td class="p-3"><div class="h-4 w-20 skeleton-loader"></div></td>
            <td class="p-3"><div class="h-4 w-24 skeleton-loader"></div></td>
            <td class="p-3">
                <div class="h-4 w-32 skeleton-loader mb-2"></div>
                <div class="h-3 w-20 skeleton-loader"></div>
            </td>
            <td class="p-3 text-right"><div class="h-4 w-24 skeleton-loader ml-auto"></div></td>
            <td class="p-3 text-center"><div class="h-4 w-20 skeleton-loader rounded-full mx-auto"></div></td>
        </tr>
    `;
}

const generateClientTxId = () => {
    if (window.crypto?.randomUUID) {
        return window.crypto.randomUUID();
    }
    return `tx_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
};

export function Transactions() {
    const view = document.createElement('div');
    view.innerHTML = `
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Forms Section -->
            <div class="lg:col-span-1 flex flex-col gap-8">
                <!-- P2P Transfer -->
                <div id="p2p-card" class="bg-surface rounded-xl shadow-md">
                    <div class="p-6 border-b border-border"><h3 class="text-lg font-semibold text-secondary-dark flex items-center gap-3">${ArrowRight} Chuyển khoản P2P</h3><p class="text-sm text-secondary mt-1">Chuyển tiền giữa hai tài khoản khách hàng.</p></div>
                    <form id="p2p-form" class="p-6 space-y-4">
                        <div><label for="sender-id" class="block text-sm font-medium text-secondary-dark mb-1">ID tài khoản người gửi</label><input type="text" id="sender-id" name="sender_id" class="w-full bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary" required></div>
                        <div><label for="receiver-id" class="block text-sm font-medium text-secondary-dark mb-1">ID tài khoản người nhận</label><input type="text" id="receiver-id" name="receiver_id" class="w-full bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary" required></div>
                        <div><label for="p2p-amount" class="block text-sm font-medium text-secondary-dark mb-1">Số tiền</label><input type="number" id="p2p-amount" name="amount" step="1000" class="w-full bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary" required></div>
                        <button type="submit" class="w-full bg-primary text-background font-bold py-2.5 rounded-lg hover:bg-primary-dark transition-colors">Gửi</button>
                        <div id="p2p-response" class="text-center text-sm mt-2 h-4"></div>
                    </form>
                </div>

                <!-- Company Treasury -->
                <div id="treasury-card" class="bg-surface rounded-xl shadow-md">
                    <div class="p-6 border-b border-border"><h3 class="text-lg font-semibold text-secondary-dark flex items-center gap-3">${Landmark} Ngân quỹ công ty</h3><p class="text-sm text-secondary mt-1">Quản lý dòng tiền cấp công ty.</p></div>
                    <form id="treasury-form" class="p-6 space-y-4">
                         <div><label for="treasury-account-id" class="block text-sm font-medium text-secondary-dark mb-1">ID tài khoản mục tiêu</label><input type="text" id="treasury-account-id" name="account_id" class="w-full bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary" required></div>
                        <div><label for="treasury-type" class="block text-sm font-medium text-secondary-dark mb-1">Loại giao dịch</label><select id="treasury-type" name="type" class="w-full bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary"><option value="cash_in">Tiền vào</option><option value="cash_out">Tiền ra</option></select></div>
                        <div><label for="treasury-amount" class="block text-sm font-medium text-secondary-dark mb-1">Số tiền</label><input type="number" id="treasury-amount" name="amount" step="1000" class="w-full bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary" required></div>
                        <button type="submit" class="w-full bg-secondary-dark text-white font-bold py-2.5 rounded-lg hover:bg-black transition-colors">Xử lý giao dịch</button>
                        <div id="treasury-response" class="text-center text-sm mt-2 h-4"></div>
                    </form>
                </div>
            </div>

            <!-- Transaction Log Section -->
            <div id="tx-log-card" class="lg:col-span-2 bg-surface rounded-xl shadow-md">
                <div class="p-6 flex justify-between items-center border-b border-border">
                    <div><h3 class="text-lg font-semibold text-secondary-dark">Giao dịch hệ thống gần đây</h3><p class="text-sm text-secondary mt-1">Nguồn cấp dữ liệu trực tiếp của tất cả các giao dịch trên toàn nền tảng.</p></div>
                    <button class="bg-secondary-light text-secondary-dark font-bold py-2 px-4 rounded-lg flex items-center gap-2 hover:bg-border">${Download}<span>Xuất</span></button>
                </div>
                <div class="overflow-x-auto"><table class="w-full text-sm text-left"><thead class="bg-secondary-light text-secondary-dark uppercase text-xs"><tr><th class="p-3">ID</th><th class="p-3">Ngày</th><th class="p-3">Mô tả</th><th class="p-3 text-right">Số tiền</th><th class="p-3 text-center">Trạng thái</th></tr></thead><tbody id="tx-log-body"></tbody></table></div>
            </div>
        </div>
    `;

    const txLogBody = view.querySelector('#tx-log-body');

    async function loadTransactions() {
        txLogBody.innerHTML = Array(5).fill(0).map(SkeletonLogRow).join('');
        try {
            const { items } = await getAllTransactions({ limit: 20 });
            txLogBody.innerHTML = items.map(TransactionLogRow).join('');
            animateStaggerIn(view.querySelectorAll('.tx-log-row'));
        } catch (error) {
            txLogBody.innerHTML = `<tr><td colspan="5" class="text-center p-8 text-danger"><p class="font-bold">Không thể tải giao dịch</p><p class="text-xs text-secondary mt-2 max-w-md mx-auto">${error.message}</p></td></tr>`;
        }
    }
    
    // Form Handlers
    const p2pForm = view.querySelector('#p2p-form');
    p2pForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(p2pForm);
        const senderId = formData.get('sender_id');
        const receiverId = formData.get('receiver_id');
        const data = {
            ...Object.fromEntries(formData.entries()),
            account_id: senderId,
            client_tx_id: generateClientTxId(),
            type: 'p2p',
            currency: 'VND',
            description: `P2P từ ${senderId} đến ${receiverId}`
        };
        const responseEl = view.querySelector('#p2p-response');
        try {
            await createTransaction(data);
            showResponseMessage(responseEl, 'Chuyển khoản thành công!', true);
            p2pForm.reset();
            loadTransactions();
        } catch (error) {
            showResponseMessage(responseEl, `Lỗi: ${error.message}`, false);
        }
    });

    const treasuryForm = view.querySelector('#treasury-form');
    treasuryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(treasuryForm);
        const data = {
            ...Object.fromEntries(formData.entries()),
            account_id: formData.get('account_id'),
            client_tx_id: generateClientTxId(),
            currency: 'VND',
            description: formData.get('type') === 'cash_in' 
                ? `Nạp tiền vào tài khoản ${formData.get('account_id')}` 
                : `Rút tiền từ tài khoản ${formData.get('account_id')}`
        };
        const responseEl = view.querySelector('#treasury-response');
        try {
            await createTransaction(data);
            showResponseMessage(responseEl, 'Giao dịch ngân quỹ thành công!', true);
            treasuryForm.reset();
            loadTransactions();
        } catch (error) {
            showResponseMessage(responseEl, `Lỗi: ${error.message}`, false);
        }
    });
    
    // Initial Load
    loadTransactions();
    setTimeout(() => animateStaggerIn(['#p2p-card', '#treasury-card', '#tx-log-card']), 10);

    return view;
}
