import { ArrowRight, Landmark, Download, Search, ArrowLeftRight } from 'lucide-static';
import { formatCurrency, formatDate } from '../utils/helpers.js';
import { createTransaction, getAllTransactions, validateAccount, getTransfers } from '../services/api.js';
import { animateStaggerIn, showResponseMessage } from '../utils/animations.js';

function TransactionLogRow(tx) {
    const isPositive = tx.amount >= 0;
    const statusMap = {
        'Completed': 'bg-green-900/50 text-success',
        'Pending': 'bg-yellow-900/50 text-warning',
        'Failed': 'bg-red-900/50 text-danger',
        'SETTLED': 'bg-green-900/50 text-success',
        'PENDING': 'bg-yellow-900/50 text-warning',
        'FAILED': 'bg-red-900/50 text-danger',
    };
    const statusValue = tx.status || 'Completed';
    const statusClass = statusMap[statusValue] || 'bg-gray-700 text-gray-400';

    // Check if this is a P2P transaction
    const extraJson = tx.extra_json || {};
    const isP2P = extraJson.p2p_role || extraJson.transfer_id;
    const transactionType = tx.transaction_type || tx.type || tx.direction || 'unknown';

    let transactionLabel = '';
    if (isP2P || transactionType === 'p2p') {
        transactionLabel = '<span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded-full bg-blue-900/50 text-blue-300">Giao dịch khách hàng</span>';
    } else if (transactionType === 'cash_in' || transactionType === 'cash_out') {
        transactionLabel = '<span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded-full bg-purple-900/50 text-purple-300">Giao dịch ngân hàng</span>';
    } else {
        transactionLabel = `<span class="ml-2 px-2 py-0.5 text-xs font-semibold rounded-full bg-gray-700 text-gray-400">${transactionType}</span>`;
    }

    const txId = tx.tx_id || tx.transaction_id || tx.id || 'N/A';
    const dateValue = tx.event_ts || tx.created_at || tx.event_date || null;
    const description = tx.description || extraJson.description || (isP2P ? `Chuyển khoản P2P` : 'Giao dịch') || 'N/A';

    return `
        <tr class="tx-log-row border-b border-border last:border-0">
            <td class="p-3">${txId}</td>
            <td class="p-3">${dateValue ? formatDate(dateValue) : 'N/A'}</td>
            <td class="p-3">
                <div class="font-medium text-secondary-dark">${description} ${transactionLabel}</div>
                <div class="text-xs text-secondary">${transactionType}</div>
            </td>
            <td class="p-3 font-mono text-right ${isPositive ? 'text-success' : 'text-danger'}">
                ${formatCurrency(tx.amount, tx.currency || 'VND')}
            </td>
            <td class="p-3 text-center">
                <span class="px-2 py-1 text-xs font-semibold rounded-full ${statusClass}">
                    ${statusValue}
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
                        <div>
                            <label for="sender-id" class="block text-sm font-medium text-secondary-dark mb-1">ID tài khoản người gửi</label>
                            <div class="flex gap-2">
                                <input type="text" id="sender-id" name="sender_id" class="flex-1 bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary" required>
                                <button type="button" id="validate-sender-btn" class="bg-secondary-light hover:bg-border text-secondary-dark px-3 rounded-lg transition-colors" title="Kiểm tra tài khoản">${Search}</button>
                            </div>
                            <div id="sender-validation" class="text-xs mt-1 h-4"></div>
                        </div>
                        <div>
                            <label for="receiver-id" class="block text-sm font-medium text-secondary-dark mb-1">ID tài khoản người nhận</label>
                            <div class="flex gap-2">
                                <input type="text" id="receiver-id" name="receiver_id" class="flex-1 bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary" required>
                                <button type="button" id="validate-receiver-btn" class="bg-secondary-light hover:bg-border text-secondary-dark px-3 rounded-lg transition-colors" title="Kiểm tra tài khoản">${Search}</button>
                            </div>
                            <div id="receiver-validation" class="text-xs mt-1 h-4"></div>
                        </div>
                        <div><label for="p2p-amount" class="block text-sm font-medium text-secondary-dark mb-1">Số tiền</label><input type="number" id="p2p-amount" name="amount" step="1000" min="0" class="w-full bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary" required></div>
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
                    <div>
                        <div class="flex items-center gap-4 mb-2">
                            <h3 class="text-lg font-semibold text-secondary-dark">Giao dịch hệ thống gần đây</h3>
                            <div class="flex gap-2">
                                <button id="show-transactions-btn" class="px-3 py-1 text-sm rounded-lg bg-primary text-background font-semibold">Giao dịch</button>
                                <button id="show-transfers-btn" class="px-3 py-1 text-sm rounded-lg bg-secondary-light text-secondary-dark hover:bg-border">Chuyển khoản</button>
                            </div>
                        </div>
                        <p class="text-sm text-secondary">Nguồn cấp dữ liệu trực tiếp của tất cả các giao dịch trên toàn nền tảng.</p>
                    </div>
                    <button class="bg-secondary-light text-secondary-dark font-bold py-2 px-4 rounded-lg flex items-center gap-2 hover:bg-border">${Download}<span>Xuất</span></button>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm text-left">
                        <thead class="bg-secondary-light text-secondary-dark uppercase text-xs">
                            <tr id="table-header"></tr>
                        </thead>
                        <tbody id="tx-log-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    const txLogBody = view.querySelector('#tx-log-body');
    const tableHeader = view.querySelector('#table-header');
    let currentView = 'transactions'; // 'transactions' or 'transfers'

    function TransferLogRow(transfer) {
        const statusMap = {
            'SETTLED': 'bg-green-900/50 text-success',
            'PENDING': 'bg-yellow-900/50 text-warning',
            'FAILED': 'bg-red-900/50 text-danger',
            'REVERSED': 'bg-gray-700 text-gray-400',
        };
        const statusClass = statusMap[transfer.status] || 'bg-gray-700 text-gray-400';

        return `
            <tr class="transfer-log-row border-b border-border last:border-0">
                <td class="p-3">${transfer.transfer_id || 'N/A'}</td>
                <td class="p-3">${formatDate(transfer.created_at)}</td>
                <td class="p-3">
                    <div class="font-medium text-secondary-dark">${transfer.from_account} ${ArrowLeftRight} ${transfer.to_account}</div>
                    <div class="text-xs text-secondary">Chuyển khoản P2P</div>
                </td>
                <td class="p-3 font-mono text-right text-primary">
                    ${formatCurrency(transfer.amount, transfer.currency || 'VND')}
                </td>
                <td class="p-3 text-center">
                    <span class="px-2 py-1 text-xs font-semibold rounded-full ${statusClass}">
                        ${transfer.status || 'SETTLED'}
                    </span>
                </td>
            </tr>
        `;
    }

    function SkeletonTransferRow() {
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

    function updateTableHeader(viewType) {
        if (viewType === 'transfers') {
            tableHeader.innerHTML = `
                <th class="p-3">ID Chuyển khoản</th>
                <th class="p-3">Ngày</th>
                <th class="p-3">Từ → Đến</th>
                <th class="p-3 text-right">Số tiền</th>
                <th class="p-3 text-center">Trạng thái</th>
            `;
        } else {
            tableHeader.innerHTML = `
                <th class="p-3">ID</th>
                <th class="p-3">Ngày</th>
                <th class="p-3">Mô tả</th>
                <th class="p-3 text-right">Số tiền</th>
                <th class="p-3 text-center">Trạng thái</th>
            `;
        }
    }

    async function loadTransactions() {
        currentView = 'transactions';
        updateTableHeader('transactions');
        txLogBody.innerHTML = Array(5).fill(0).map(SkeletonLogRow).join('');
        try {
            const { items } = await getAllTransactions({ limit: 20 });
            txLogBody.innerHTML = items.map(TransactionLogRow).join('');
            animateStaggerIn(view.querySelectorAll('.tx-log-row'));
        } catch (error) {
            txLogBody.innerHTML = `<tr><td colspan="5" class="text-center p-8 text-danger"><p class="font-bold">Không thể tải giao dịch</p><p class="text-xs text-secondary mt-2 max-w-md mx-auto">${error.message}</p></td></tr>`;
        }
    }

    async function loadTransfers() {
        currentView = 'transfers';
        updateTableHeader('transfers');
        txLogBody.innerHTML = Array(5).fill(0).map(SkeletonTransferRow).join('');
        try {
            console.log('Loading transfers...');
            const response = await getTransfers({ limit: 20 });
            console.log('Transfers response:', response);
            const items = response.items || response.data || [];
            if (items && items.length > 0) {
                console.log(`Found ${items.length} transfers`);
                txLogBody.innerHTML = items.map(TransferLogRow).join('');
                animateStaggerIn(view.querySelectorAll('.transfer-log-row'));
            } else {
                console.log('No transfers found');
                txLogBody.innerHTML = `<tr><td colspan="5" class="text-center p-8 text-secondary"><p class="font-bold">Chưa có chuyển khoản nào</p><p class="text-xs text-secondary mt-2">Các giao dịch chuyển khoản P2P sẽ hiển thị ở đây</p></td></tr>`;
            }
        } catch (error) {
            console.error('Error loading transfers:', error);
            txLogBody.innerHTML = `<tr><td colspan="5" class="text-center p-8 text-danger"><p class="font-bold">Không thể tải chuyển khoản</p><p class="text-xs text-secondary mt-2 max-w-md mx-auto">${error.message}</p></td></tr>`;
        }
    }
    
    // Account validation handlers
    async function validateAccountId(accountId, validationEl) {
        if (!accountId) {
            validationEl.textContent = '';
            return false;
        }
        validationEl.textContent = 'Đang kiểm tra...';
        validationEl.className = 'text-xs mt-1 h-4 text-secondary';
        try {
            const account = await validateAccount(accountId);
            validationEl.textContent = `✓ Tài khoản hợp lệ (${account.account?.status || 'N/A'})`;
            validationEl.className = 'text-xs mt-1 h-4 text-success';
            return true;
        } catch (error) {
            validationEl.textContent = `✗ ${error.message}`;
            validationEl.className = 'text-xs mt-1 h-4 text-danger';
            return false;
        }
    }

    const senderInput = view.querySelector('#sender-id');
    const receiverInput = view.querySelector('#receiver-id');
    const senderValidation = view.querySelector('#sender-validation');
    const receiverValidation = view.querySelector('#receiver-validation');
    const validateSenderBtn = view.querySelector('#validate-sender-btn');
    const validateReceiverBtn = view.querySelector('#validate-receiver-btn');

    validateSenderBtn.addEventListener('click', async () => {
        await validateAccountId(senderInput.value, senderValidation);
    });

    validateReceiverBtn.addEventListener('click', async () => {
        await validateAccountId(receiverInput.value, receiverValidation);
    });

    // Auto-validate on blur
    senderInput.addEventListener('blur', async () => {
        if (senderInput.value) {
            await validateAccountId(senderInput.value, senderValidation);
        }
    });

    receiverInput.addEventListener('blur', async () => {
        if (receiverInput.value) {
            await validateAccountId(receiverInput.value, receiverValidation);
        }
    });

    // Form Handlers
    const p2pForm = view.querySelector('#p2p-form');
    p2pForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(p2pForm);
        const senderId = formData.get('sender_id');
        const receiverId = formData.get('receiver_id');
        const amount = parseFloat(formData.get('amount'));
        
        const responseEl = view.querySelector('#p2p-response');
        
        // Validate accounts before submitting
        const senderValid = await validateAccountId(senderId, senderValidation);
        const receiverValid = await validateAccountId(receiverId, receiverValidation);
        
        if (!senderValid || !receiverValid) {
            showResponseMessage(responseEl, 'Vui lòng kiểm tra và đảm bảo cả hai tài khoản đều tồn tại.', false);
            return;
        }
        
        if (senderId === receiverId) {
            showResponseMessage(responseEl, 'Không thể chuyển tiền cho chính mình. Vui lòng chọn tài khoản người nhận khác.', false);
            return;
        }
        
        if (amount <= 0) {
            showResponseMessage(responseEl, 'Số tiền phải lớn hơn 0.', false);
            return;
        }
        
        const data = {
            sender_id: senderId,
            receiver_id: receiverId,
            amount: amount,
            client_tx_id: generateClientTxId(),
            currency: 'VND',
            description: `Chuyển khoản P2P từ ${senderId} đến ${receiverId}`
        };
        
        try {
            await createTransaction(data);
            showResponseMessage(responseEl, 'Chuyển khoản thành công!', true);
            p2pForm.reset();
            senderValidation.textContent = '';
            receiverValidation.textContent = '';
            // Reload current view
            if (currentView === 'transfers') {
                loadTransfers();
            } else {
                loadTransactions();
            }
        } catch (error) {
            showResponseMessage(responseEl, `Lỗi: ${error.message}`, false);
        }
    });

    const treasuryForm = view.querySelector('#treasury-form');
    treasuryForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(treasuryForm);
        const accountId = formData.get('account_id');
        const transactionType = formData.get('type'); // 'cash_in' or 'cash_out'
        const amountStr = formData.get('amount');
        
        // Validate required fields
        if (!accountId || !transactionType || !amountStr) {
            const responseEl = view.querySelector('#treasury-response');
            showResponseMessage(responseEl, 'Vui lòng điền đầy đủ thông tin!', false);
            return;
        }
        
        // Convert amount to number
        const amount = parseFloat(amountStr);
        if (isNaN(amount) || amount <= 0) {
            const responseEl = view.querySelector('#treasury-response');
            showResponseMessage(responseEl, 'Số tiền không hợp lệ!', false);
            return;
        }
        
        // Prepare data for API - map 'type' to 'transaction_type'
        const data = {
            account_id: accountId,
            transaction_type: transactionType, // Map 'type' to 'transaction_type'
            amount: amount,
            currency: 'VND',
            client_tx_id: generateClientTxId(),
            description: transactionType === 'cash_in' 
                ? `Nạp tiền vào tài khoản ${accountId}` 
                : `Rút tiền từ tài khoản ${accountId}`,
            merchant: 'COMPANY_TREASURY',
            status: 'SETTLED',
            extra_json: {
                form_type: 'treasury',
                source: 'company_treasury_management'
            }
        };
        
        const responseEl = view.querySelector('#treasury-response');
        try {
            const result = await createTransaction(data);
            console.log('Treasury transaction result:', result);
            
            if (result && result.status === 'success') {
                showResponseMessage(responseEl, 'Giao dịch ngân quỹ thành công!', true);
                treasuryForm.reset();
                loadTransactions();
            } else {
                showResponseMessage(responseEl, `Lỗi: ${result?.error || 'Không thể xử lý giao dịch'}`, false);
            }
        } catch (error) {
            console.error('Treasury transaction error:', error);
            showResponseMessage(responseEl, `Lỗi: ${error.message}`, false);
        }
    });
    
    // View toggle buttons
    const showTransactionsBtn = view.querySelector('#show-transactions-btn');
    const showTransfersBtn = view.querySelector('#show-transfers-btn');
    
    showTransactionsBtn.addEventListener('click', () => {
        showTransactionsBtn.className = 'px-3 py-1 text-sm rounded-lg bg-primary text-background font-semibold';
        showTransfersBtn.className = 'px-3 py-1 text-sm rounded-lg bg-secondary-light text-secondary-dark hover:bg-border';
        loadTransactions();
    });
    
    showTransfersBtn.addEventListener('click', () => {
        showTransfersBtn.className = 'px-3 py-1 text-sm rounded-lg bg-primary text-background font-semibold';
        showTransactionsBtn.className = 'px-3 py-1 text-sm rounded-lg bg-secondary-light text-secondary-dark hover:bg-border';
        loadTransfers();
    });

    // Initial Load
    loadTransactions();
    setTimeout(() => animateStaggerIn(['#p2p-card', '#treasury-card', '#tx-log-card']), 10);

    return view;
}
