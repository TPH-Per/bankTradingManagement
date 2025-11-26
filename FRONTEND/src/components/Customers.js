import { Plus, Save, User, Hash, Mail, Calendar, ShieldCheck, BarChart, ArrowLeftRight, DollarSign, FileText } from 'lucide-static';
import { searchCustomers, getCustomerById, getTransactions, getCustomerStats, updateCustomer, createCustomer, getAccountBalance, updateAccountBalance, getAccountStatement } from '../services/api.js';
import { formatDate, formatCurrency } from '../utils/helpers.js';
import { Modal } from './Modal.js';
import { animateStaggerIn, showResponseMessage } from '../utils/animations.js';

function CustomerRow(customer) {
    const statusMap = {
        'ACTIVE': 'bg-green-900/50 text-success',
        'active': 'bg-green-900/50 text-success',
        'LOCKED': 'bg-gray-700 text-gray-400',
        'inactive': 'bg-gray-700 text-gray-400',
        'suspended': 'bg-yellow-900/50 text-warning',
    };
    const statusClass = statusMap[customer.status] || statusMap['inactive'];

    return `
        <tr class="customer-row border-b border-border hover:bg-secondary-light">
            <td class="p-4 font-mono text-primary">${customer.account_id || '-'}</td>
            <td class="p-4">
                <div class="font-medium text-secondary-dark">${customer.full_name || '-'}</div>
            </td>
            <td class="p-4">${customer.national_id || '-'}</td>
            <td class="p-4 text-sm">${customer.email || '-'}</td>
            <td class="p-4">${customer.phone || '-'}</td>
            <td class="p-4">
                <span class="px-2 py-1 text-xs font-semibold rounded-full ${statusClass}">
                    ${customer.status}
                </span>
            </td>
            <td class="p-4 text-right">
                <button data-id="${customer.account_id}" class="view-customer-btn text-primary hover:underline">Xem</button>
            </td>
        </tr>
    `;
}

function SkeletonRow() {
    return `
        <tr class="border-b border-border">
            <td class="p-4"><div class="h-4 w-20 skeleton-loader"></div></td>
            <td class="p-4">
                <div class="h-4 w-32 skeleton-loader mb-2"></div>
                <div class="h-3 w-40 skeleton-loader"></div>
            </td>
            <td class="p-4"><div class="h-4 w-24 skeleton-loader"></div></td>
            <td class="p-4"><div class="h-4 w-24 skeleton-loader"></div></td>
            <td class="p-4"><div class="h-4 w-20 skeleton-loader rounded-full"></div></td>
            <td class="p-4"><div class="h-4 w-24 skeleton-loader"></div></td>
            <td class="p-4 text-right"><div class="h-4 w-12 skeleton-loader inline-block"></div></td>
        </tr>
    `;
}

function createDetailModalContent() {
    const content = document.createElement('div');
    content.innerHTML = `
        <div class="flex border-b border-border">
            <button data-tab="profile" class="detail-tab-btn active p-3 font-medium flex items-center gap-2">${User} Hồ sơ</button>
            <button data-tab="balance" class="detail-tab-btn p-3 font-medium flex items-center gap-2">${DollarSign} Số dư</button>
            <button data-tab="statement" class="detail-tab-btn p-3 font-medium flex items-center gap-2">${FileText} Sao kê</button>
            <button data-tab="transactions" class="detail-tab-btn p-3 font-medium flex items-center gap-2">${ArrowLeftRight} Giao dịch</button>
            <button data-tab="statistics" class="detail-tab-btn p-3 font-medium flex items-center gap-2">${BarChart} Thống kê</button>
        </div>
        <div class="mt-4 min-h-[300px]">
            <div id="profile-tab" class="detail-tab-content"></div>
            <div id="balance-tab" class="detail-tab-content hidden"></div>
            <div id="statement-tab" class="detail-tab-content hidden"></div>
            <div id="transactions-tab" class="detail-tab-content hidden"></div>
            <div id="statistics-tab" class="detail-tab-content hidden"></div>
        </div>
    `;
    return content;
}

function renderProfileTab(container, customer) {
    container.innerHTML = `
        <form id="customer-update-form" class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div><label class="block text-sm font-medium text-secondary-dark mb-1">Họ và tên</label><input type="text" name="full_name" value="${customer.full_name}" class="w-full bg-background border-border rounded-lg p-2"></div>
                <div><label class="block text-sm font-medium text-secondary-dark mb-1">Email</label><input type="email" name="email" value="${customer.email}" class="w-full bg-background border-border rounded-lg p-2"></div>
            </div>
            <div><label class="block text-sm font-medium text-secondary-dark mb-1">CMND/CCCD</label><input type="text" name="national_id" value="${customer.national_id}" class="w-full bg-background border-border rounded-lg p-2"></div>
            <div><label class="block text-sm font-medium text-secondary-dark mb-1">Trạng thái</label>
                <select name="status" class="w-full bg-background border-border rounded-lg p-2">
                    <option value="active" ${customer.status === 'active' ? 'selected' : ''}>Hoạt động</option>
                    <option value="inactive" ${customer.status === 'inactive' ? 'selected' : ''}>Không hoạt động</option>
                    <option value="suspended" ${customer.status === 'suspended' ? 'selected' : ''}>Bị khóa</option>
                </select>
            </div>
            <div class="flex justify-end pt-4">
                <button type="submit" class="bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg flex items-center gap-2">${Save} <span>Lưu thay đổi</span></button>
            </div>
            <div id="update-response" class="text-center text-sm mt-2"></div>
        </form>
    `;

    const form = container.querySelector('#customer-update-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        const responseEl = container.querySelector('#update-response');
        try {
            await updateCustomer(customer.account_id, data);
            showResponseMessage(responseEl, 'Cập nhật khách hàng thành công!', true);
        } catch (error) {
            showResponseMessage(responseEl, `Lỗi: ${error.message}`, false);
        }
    });
}

async function renderTransactionsTab(container, accountId) {
    container.innerHTML = `<div class="text-center p-8 text-secondary">Đang tải giao dịch...</div>`;
    try {
        const { items } = await getTransactions({ account_id: accountId, limit: 10 });
        if (!items || items.length === 0) {
            container.innerHTML = `<div class="text-center p-8 text-secondary">Không tìm thấy giao dịch nào cho khách hàng này.</div>`;
            return;
        }
        container.innerHTML = `
            <table class="w-full text-sm">
                <thead><tr class="border-b border-border"><th class="p-2 text-left">Ngày</th><th class="p-2 text-left">Mô tả</th><th class="p-2 text-right">Số tiền</th></tr></thead>
                <tbody>${items.map(tx => `
                    <tr class="border-b border-border last:border-0">
                        <td class="p-2">${formatDate(tx.created_at)}</td>
                        <td class="p-2">${tx.description}</td>
                        <td class="p-2 text-right font-mono ${tx.amount >= 0 ? 'text-success' : 'text-danger'}">${formatCurrency(tx.amount, 'VND')}</td>
                    </tr>
                `).join('')}</tbody>
            </table>
        `;
    } catch (error) {
        container.innerHTML = `<div class="text-center p-8 text-danger"><p class="font-bold">Không thể tải giao dịch</p><p class="text-xs text-secondary mt-2">${error.message}</p></div>`;
    }
}

async function renderBalanceTab(container, accountId, onBalanceUpdate = null) {
    container.innerHTML = `<div class="text-center p-8 text-secondary">Đang tải số dư...</div>`;
    try {
        const balance = await getAccountBalance(accountId);
        const balanceValue = balance.balance || 0;
        const balanceClass = balanceValue >= 0 ? 'text-success' : 'text-danger';

        container.innerHTML = `
            <div class="space-y-6">
                <div class="bg-secondary-light p-6 rounded-lg text-center">
                    <p class="text-sm text-secondary mb-2">Số dư hiện tại</p>
                    <p class="text-4xl font-bold ${balanceClass}">${formatCurrency(balanceValue, 'VND')}</p>
                    ${balance.updated_at ? `<p class="text-xs text-secondary mt-2">Cập nhật: ${formatDate(balance.updated_at)}</p>` : ''}
                </div>
                
                <div class="border-t border-border pt-4">
                    <h3 class="font-semibold text-secondary-dark mb-4">Cập nhật số dư</h3>
                    <form id="balance-update-form" class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium text-secondary-dark mb-1">Thao tác</label>
                            <select name="operation" class="w-full bg-background border-border rounded-lg p-2" required>
                                <option value="add">Thêm tiền (+)</option>
                                <option value="subtract">Trừ tiền (-)</option>
                                <option value="set">Đặt số dư (=)</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-secondary-dark mb-1">Số tiền</label>
                            <input type="number" name="amount" step="0.01" min="0" required class="w-full bg-background border-border rounded-lg p-2" placeholder="Nhập số tiền">
                        </div>
                        <button type="submit" class="w-full bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg flex items-center justify-center gap-2">
                            ${Save} <span>Cập nhật số dư</span>
                        </button>
                        <div id="balance-update-response" class="text-center text-sm mt-2"></div>
                    </form>
                </div>
            </div>
        `;

        const form = container.querySelector('#balance-update-form');
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            const data = {
                amount: parseFloat(formData.get('amount')),
                operation: formData.get('operation')
            };
            const responseEl = container.querySelector('#balance-update-response');
            try {
                const result = await updateAccountBalance(accountId, data);
                console.log('Balance update result:', result);
                const newBalance = result.balance || 0;
                showResponseMessage(responseEl, `Cập nhật số dư thành công! Số dư mới: ${formatCurrency(newBalance, 'VND')}`, true);
                // Reload balance immediately
                setTimeout(async () => {
                    await renderBalanceTab(container, accountId, onBalanceUpdate);
                }, 500);
                // Refresh customer list after a short delay to ensure balance is saved
                if (onBalanceUpdate) {
                    setTimeout(() => {
                        onBalanceUpdate();
                    }, 1500);
                }
            } catch (error) {
                showResponseMessage(responseEl, `Lỗi: ${error.message}`, false);
            }
        });
    } catch (error) {
        container.innerHTML = `<div class="text-center p-8 text-danger"><p class="font-bold">Không thể tải số dư</p><p class="text-xs text-secondary mt-2">${error.message}</p></div>`;
    }
}

async function renderStatementTab(container, accountId) {
    // Default to last 30 days
    const today = new Date();
    const thirtyDaysAgo = new Date(today);
    thirtyDaysAgo.setDate(today.getDate() - 30);

    const defaultDateFrom = thirtyDaysAgo.toISOString().split('T')[0];
    const defaultDateTo = today.toISOString().split('T')[0];

    container.innerHTML = `
        <div class="space-y-4">
            <div class="flex gap-4 items-end">
                <div class="flex-1">
                    <label class="block text-sm font-medium text-secondary-dark mb-1">Từ ngày</label>
                    <input type="date" id="statement-date-from" value="${defaultDateFrom}" class="w-full bg-background border-border rounded-lg p-2">
                </div>
                <div class="flex-1">
                    <label class="block text-sm font-medium text-secondary-dark mb-1">Đến ngày</label>
                    <input type="date" id="statement-date-to" value="${defaultDateTo}" class="w-full bg-background border-border rounded-lg p-2">
                </div>
                <button id="load-statement-btn" class="bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg">Tải sao kê</button>
            </div>
            <div id="statement-content" class="mt-4">
                <div class="text-center p-8 text-secondary">Chọn khoảng thời gian và nhấn "Tải sao kê"</div>
            </div>
        </div>
    `;

    const loadBtn = container.querySelector('#load-statement-btn');
    const dateFromInput = container.querySelector('#statement-date-from');
    const dateToInput = container.querySelector('#statement-date-to');
    const contentDiv = container.querySelector('#statement-content');

    async function loadStatement() {
        const dateFrom = dateFromInput.value;
        const dateTo = dateToInput.value;

        if (!dateFrom || !dateTo) {
            showResponseMessage(contentDiv, 'Vui lòng chọn cả hai ngày', false);
            return;
        }

        if (new Date(dateFrom) > new Date(dateTo)) {
            showResponseMessage(contentDiv, 'Ngày bắt đầu phải nhỏ hơn ngày kết thúc', false);
            return;
        }

        contentDiv.innerHTML = `<div class="text-center p-8 text-secondary">Đang tải sao kê...</div>`;

        try {
            const statement = await getAccountStatement(accountId, { date_from: dateFrom, date_to: dateTo });
            const snapshots = statement.snapshots || [];

            if (snapshots.length === 0) {
                contentDiv.innerHTML = `
                    <div class="text-center p-8 text-secondary">
                        <p class="font-bold">Không có dữ liệu sao kê</p>
                        <p class="text-xs text-secondary mt-2">Không tìm thấy bản ghi nào trong khoảng thời gian đã chọn</p>
                    </div>
                `;
                return;
            }

            // Calculate totals
            const totalDebit = snapshots.reduce((sum, s) => sum + (s.total_debit || 0), 0);
            const totalCredit = snapshots.reduce((sum, s) => sum + (s.total_credit || 0), 0);
            const totalTx = snapshots.reduce((sum, s) => sum + (s.num_tx || 0), 0);

            contentDiv.innerHTML = `
                <div class="space-y-4">
                    <div class="bg-secondary-light p-4 rounded-lg">
                        <div class="grid grid-cols-3 gap-4 text-center">
                            <div>
                                <p class="text-lg font-bold text-secondary-dark">${formatCurrency(totalDebit, 'VND')}</p>
                                <p class="text-xs text-secondary">Tổng chi</p>
                            </div>
                            <div>
                                <p class="text-lg font-bold text-secondary-dark">${formatCurrency(totalCredit, 'VND')}</p>
                                <p class="text-xs text-secondary">Tổng thu</p>
                            </div>
                            <div>
                                <p class="text-lg font-bold text-secondary-dark">${totalTx}</p>
                                <p class="text-xs text-secondary">Tổng giao dịch</p>
                            </div>
                        </div>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead class="bg-secondary-light text-secondary-dark uppercase text-xs">
                                <tr>
                                    <th class="p-3 text-left">Ngày</th>
                                    <th class="p-3 text-right">Số dư đầu ngày</th>
                                    <th class="p-3 text-right">Tổng thu</th>
                                    <th class="p-3 text-right">Tổng chi</th>
                                    <th class="p-3 text-right">Số dư cuối ngày</th>
                                    <th class="p-3 text-center">Giao dịch</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${snapshots.map(snapshot => `
                                    <tr class="border-b border-border hover:bg-secondary-light">
                                        <td class="p-3">${formatDate(snapshot.day)}</td>
                                        <td class="p-3 text-right font-mono">${formatCurrency(snapshot.balance_open || 0, 'VND')}</td>
                                        <td class="p-3 text-right font-mono text-success">${formatCurrency(snapshot.total_credit || 0, 'VND')}</td>
                                        <td class="p-3 text-right font-mono text-danger">${formatCurrency(snapshot.total_debit || 0, 'VND')}</td>
                                        <td class="p-3 text-right font-mono font-bold">${formatCurrency(snapshot.balance_close || 0, 'VND')}</td>
                                        <td class="p-3 text-center">${snapshot.num_tx || 0}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        } catch (error) {
            contentDiv.innerHTML = `
                <div class="text-center p-8 text-danger">
                    <p class="font-bold">Không thể tải sao kê</p>
                    <p class="text-xs text-secondary mt-2">${error.message}</p>
                </div>
            `;
        }
    }

    loadBtn.addEventListener('click', loadStatement);
    // Auto-load on tab open
    loadStatement();
}

async function renderStatisticsTab(container, accountId) {
    container.innerHTML = `<div class="text-center p-8 text-secondary">Đang tải thống kê...</div>`;
    try {
        const stats = await getCustomerStats(accountId, { period: 'month' });
        container.innerHTML = `
            <div class="grid grid-cols-2 gap-4 text-center">
                <div class="bg-secondary-light p-4 rounded-lg"><p class="text-2xl font-bold text-secondary-dark">${formatCurrency(stats.total_in, 'VND')}</p><p class="text-sm text-secondary">Tổng tiền vào (Tháng)</p></div>
                <div class="bg-secondary-light p-4 rounded-lg"><p class="text-2xl font-bold text-secondary-dark">${formatCurrency(stats.total_out, 'VND')}</p><p class="text-sm text-secondary">Tổng tiền ra (Tháng)</p></div>
                <div class="bg-secondary-light p-4 rounded-lg"><p class="text-2xl font-bold text-secondary-dark">${stats.transaction_count}</p><p class="text-sm text-secondary">Giao dịch (Tháng)</p></div>
                <div class="bg-secondary-light p-4 rounded-lg"><p class="text-2xl font-bold text-secondary-dark">${formatCurrency(stats.balance, 'VND')}</p><p class="text-sm text-secondary">Số dư hiện tại</p></div>
            </div>
        `;
    } catch (error) {
        container.innerHTML = `<div class="text-center p-8 text-danger"><p class="font-bold">Không thể tải thống kê</p><p class="text-xs text-secondary mt-2">${error.message}</p></div>`;
    }
}


async function showCustomerDetail(accountId, onUpdate) {
    const modalContent = createDetailModalContent();
    const modal = Modal({ title: `Chi tiết khách hàng`, content: modalContent, size: 'max-w-3xl', onClose: onUpdate });

    const profileTab = modalContent.querySelector('#profile-tab');
    profileTab.innerHTML = `<div class="text-center p-8 text-secondary">Đang tải khách hàng...</div>`;

    // Store reference to loadCustomers function for balance updates
    let balanceUpdateCallback = null;

    try {
        const customer = await getCustomerById(accountId);
        renderProfileTab(profileTab, customer);

        modalContent.querySelectorAll('.detail-tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                modalContent.querySelectorAll('.detail-tab-btn').forEach(b => b.classList.remove('active'));
                modalContent.querySelectorAll('.detail-tab-content').forEach(c => c.classList.add('hidden'));
                btn.classList.add('active');
                const tabContainer = modalContent.querySelector(`#${btn.dataset.tab}-tab`);
                tabContainer.classList.remove('hidden');

                // Load content on demand
                if (btn.dataset.tab === 'balance') {
                    // Always reload balance tab to get latest balance
                    renderBalanceTab(tabContainer, accountId, () => {
                        if (onUpdate) onUpdate();
                    });
                } else if (btn.dataset.tab === 'transactions' && !tabContainer.hasChildNodes()) {
                    renderTransactionsTab(tabContainer, accountId);
                } else if (btn.dataset.tab === 'statistics' && !tabContainer.hasChildNodes()) {
                    renderStatisticsTab(tabContainer, accountId);
                }
            });
        });

    } catch (error) {
        profileTab.innerHTML = `<div class="text-center p-8 text-danger"><p class="font-bold">Không thể tải chi tiết khách hàng</p><p class="text-xs text-secondary mt-2">${error.message}</p></div>`;
    }
}

function showNewCustomerModal(onUpdate) {
    const formContent = document.createElement('form');
    formContent.id = 'new-customer-form';
    formContent.className = 'space-y-4';
    formContent.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div><label class="block text-sm font-medium text-secondary-dark mb-1">Họ và tên</label><input type="text" name="full_name" required class="w-full bg-background border-border rounded-lg p-2"></div>
            <div><label class="block text-sm font-medium text-secondary-dark mb-1">Email</label><input type="email" name="email" required class="w-full bg-background border-border rounded-lg p-2"></div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div><label class="block text-sm font-medium text-secondary-dark mb-1">CMND/CCCD</label><input type="text" name="national_id" required class="w-full bg-background border-border rounded-lg p-2"></div>
            <div><label class="block text-sm font-medium text-secondary-dark mb-1">Số điện thoại</label><input type="tel" name="phone" placeholder="0987654321" class="w-full bg-background border-border rounded-lg p-2"></div>
        </div>
        <div class="bg-secondary-light/30 p-3 rounded-lg">
            <p class="text-xs text-secondary-dark">
                <strong>Lưu ý:</strong> Số tài khoản sẽ được tự động tạo với định dạng <code class="bg-background px-2 py-1 rounded">000XXXX</code>
            </p>
        </div>
        <div class="flex justify-end pt-4">
            <button type="submit" class="bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg flex items-center gap-2">${Plus} <span>Tạo khách hàng</span></button>
        </div>
        <div id="create-response" class="text-center text-sm mt-2"></div>
    `;

    const modal = Modal({ title: 'Tạo khách hàng mới', content: formContent, size: 'max-w-xl' });

    formContent.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(formContent);
        const data = Object.fromEntries(formData.entries());
        const responseEl = formContent.querySelector('#create-response');

        try {
            const response = await createCustomer(data);

            // Show detailed success message with all customer info
            const account = response.account || {};
            const accountId = account.account_id || 'N/A';
            const extra = account.extra_json || {};

            const successMessage = `
                <div class="bg-success/10 border border-success rounded-lg p-4">
                    <div class="text-success font-bold text-lg mb-3 flex items-center gap-2">
                        <span style="font-size: 24px;">✅</span>
                        <span>Tạo khách hàng thành công!</span>
                    </div>
                    <div class="space-y-2 text-sm text-secondary-dark">
                        <div class="flex justify-between">
                            <span class="font-medium">Số tài khoản:</span>
                            <code class="bg-primary/20 px-3 py-1 rounded font-mono font-bold text-primary">${accountId}</code>
                        </div>
                        <div class="flex justify-between">
                            <span class="font-medium">Họ và tên:</span>
                            <span class="font-semibold">${extra.full_name || data.full_name || 'N/A'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="font-medium">CMND/CCCD:</span>
                            <span>${extra.national_id || data.national_id || 'N/A'}</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="font-medium">Email:</span>
                            <span>${extra.email || data.email || 'N/A'}</span>
                        </div>
                        ${data.phone ? `
                        <div class="flex justify-between">
                            <span class="font-medium">Số điện thoại:</span>
                            <span>${data.phone}</span>
                        </div>
                        ` : ''}
                        <div class="flex justify-between">
                            <span class="font-medium">Trạng thái:</span>
                            <span class="text-success font-semibold">${account.status || 'ACTIVE'}</span>
                        </div>
                    </div>
                    <div class="mt-3 pt-3 border-t border-success/30 text-xs text-center text-secondary">
                        Modal sẽ tự động đóng sau 5 giây...
                    </div>
                </div>
            `;
            responseEl.innerHTML = successMessage;

            // Log to console for debugging
            console.log('✅ Customer created successfully:', response);

            // Close modal after 5 seconds
            setTimeout(() => {
                modal.close();
                if (response.account && response.account.account_id) {
                    onUpdate(response.account.account_id);
                } else {
                    onUpdate();
                }
            }, 5000);
        } catch (error) {
            console.error('Create customer error:', error);
            showResponseMessage(responseEl, `Lỗi: ${error.message}`, false);
        }
    });
}


export function Customers() {
    const view = document.createElement('div');
    view.innerHTML = `
        <style>
            .detail-tab-btn.active { color: hsl(160, 80%, 45%); border-bottom: 2px solid hsl(160, 80%, 45%); }
            .search-input:focus { border-color: hsl(160, 80%, 45%); ring: 2px solid hsl(160, 80%, 45%, 0.2); }
        </style>
        <div class="flex justify-between items-center mb-6">
            <h2 class="text-2xl font-bold text-secondary-dark">Quản lý khách hàng</h2>
            <button id="new-customer-btn" class="bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg flex items-center gap-2">
                ${Plus}
                <span>Khách hàng mới</span>
            </button>
        </div>
        
        <!-- Search Section -->
        <div class="bg-surface rounded-xl shadow-md p-6 mb-6">
            <h3 class="text-lg font-semibold text-secondary-dark mb-4">Tra cứu khách hàng</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-secondary-dark mb-2">Số điện thoại</label>
                    <div class="flex gap-2">
                        <input type="text" id="search-phone" placeholder="0987654321" 
                               class="search-input flex-1 bg-background border border-border rounded-lg px-4 py-2 focus:outline-none">
                        <button id="btn-search-phone" class="bg-primary hover:bg-primary-dark text-background px-4 py-2 rounded-lg font-medium">
                            Tìm
                        </button>
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-secondary-dark mb-2">Email</label>
                    <div class="flex gap-2">
                        <input type="text" id="search-email" placeholder="example@email.com" 
                               class="search-input flex-1 bg-background border border-border rounded-lg px-4 py-2 focus:outline-none">
                        <button id="btn-search-email" class="bg-primary hover:bg-primary-dark text-background px-4 py-2 rounded-lg font-medium">
                            Tìm
                        </button>
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-secondary-dark mb-2">CMND/CCCD</label>
                    <div class="flex gap-2">
                        <input type="text" id="search-identity" placeholder="001234567890" 
                               class="search-input flex-1 bg-background border border-border rounded-lg px-4 py-2 focus:outline-none">
                        <button id="btn-search-identity" class="bg-primary hover:bg-primary-dark text-background px-4 py-2 rounded-lg font-medium">
                            Tìm
                        </button>
                    </div>
                </div>
                <div>
                    <label class="block text-sm font-medium text-secondary-dark mb-2">Mã tài khoản</label>
                    <div class="flex gap-2">
                        <input type="text" id="search-account" placeholder="0001234" 
                               class="search-input flex-1 bg-background border border-border rounded-lg px-4 py-2 focus:outline-none">
                        <button id="btn-search-account" class="bg-primary hover:bg-primary-dark text-background px-4 py-2 rounded-lg font-medium">
                            Tìm
                        </button>
                    </div>
                </div>
            </div>
            <div class="mt-4 flex justify-end">
                <button id="btn-clear-search" class="text-secondary hover:text-secondary-dark font-medium text-sm">
                    Xóa bộ lọc & hiển thị tất cả
                </button>
            </div>
        </div>
        
        <div class="bg-surface rounded-xl shadow-md overflow-hidden">
            <div class="overflow-x-auto">
                <table class="w-full text-sm text-left">
                    <thead class="bg-secondary-light text-secondary-dark uppercase text-xs">
                        <tr>
                            <th class="p-4">ID Tài khoản</th><th class="p-4">Khách hàng</th><th class="p-4">CMND/CCCD</th>
                            <th class="p-4">Email</th><th class="p-4">Số điện thoại</th>
                            <th class="p-4">Trạng thái</th><th class="p-4 text-right">Hành động</th>
                        </tr>
                    </thead>
                    <tbody id="customer-table-body"></tbody>
                </table>
            </div>
        </div>
    `;

    const tableBody = view.querySelector('#customer-table-body');

    async function performSearch(query, searchType) {
        if (!query || !query.trim()) {
            alert('Vui lòng nhập thông tin cần tìm');
            return;
        }

        tableBody.innerHTML = Array(3).fill(0).map(SkeletonRow).join('');
        try {
            const { items } = await searchCustomers({ query: query.trim(), search_type: searchType });

            if (items.length === 0) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="7" class="text-center p-8 text-secondary">
                            <p class="font-bold">Không tìm thấy khách hàng</p>
                            <p class="text-xs mt-2">Không có kết quả cho "${query}"</p>
                        </td>
                    </tr>
                `;
                return;
            }

            // Load balance for each customer
            const customersWithBalance = await Promise.all(
                items.map(async (customer) => {
                    try {
                        const balance = await getAccountBalance(customer.account_id);
                        return { ...customer, balance: balance.balance || 0 };
                    } catch (error) {
                        console.warn(`Failed to load balance for ${customer.account_id}:`, error);
                        return { ...customer, balance: 0 };
                    }
                })
            );

            tableBody.innerHTML = customersWithBalance.map(CustomerRow).join('');
            animateStaggerIn(view.querySelectorAll('.customer-row'));
        } catch (error) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center p-8 text-danger">
                        <p class="font-bold">Lỗi tìm kiếm</p>
                        <p class="text-xs text-secondary mt-2">${error.message}</p>
                    </td>
                </tr>
            `;
        }
    }

    async function loadAllCustomers() {
        tableBody.innerHTML = Array(5).fill(0).map(SkeletonRow).join('');
        try {
            // Search with empty constraints to get all
            const { items } = await searchCustomers({ query: '', search_type: 'all' });

            if (items.length === 0) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="7" class="text-center p-8 text-secondary">
                            <p class="font-bold">Chưa có khách hàng</p>
                            <p class="text-xs mt-2">Nhấn "Khách hàng mới" để tạo khách hàng đầu tiên</p>
                        </td>
                    </tr>
                `;
                return;
            }

            // Load balance for each customer
            const customersWithBalance = await Promise.all(
                items.map(async (customer) => {
                    try {
                        const balance = await getAccountBalance(customer.account_id);
                        return { ...customer, balance: balance.balance || 0 };
                    } catch (error) {
                        console.warn(`Failed to load balance for ${customer.account_id}:`, error);
                        return { ...customer, balance: 0 };
                    }
                })
            );
            tableBody.innerHTML = customersWithBalance.map(CustomerRow).join('');
            animateStaggerIn(view.querySelectorAll('.customer-row'));
        } catch (error) {
            tableBody.innerHTML = `<tr><td colspan="7" class="text-center p-8 text-danger"><p class="font-bold">Không thể tải khách hàng</p><p class="text-xs text-secondary mt-2 max-w-md mx-auto">${error.message}</p></td></tr>`;
        }
    }

    loadAllCustomers();

    // Event listeners for search buttons
    view.querySelector('#btn-search-phone').addEventListener('click', () => {
        const query = view.querySelector('#search-phone').value;
        performSearch(query, 'phone');
    });

    view.querySelector('#btn-search-email').addEventListener('click', () => {
        const query = view.querySelector('#search-email').value;
        performSearch(query, 'email');
    });

    view.querySelector('#btn-search-identity').addEventListener('click', () => {
        const query = view.querySelector('#search-identity').value;
        performSearch(query, 'identity');
    });

    view.querySelector('#btn-search-account').addEventListener('click', () => {
        const query = view.querySelector('#search-account').value;
        performSearch(query, 'account');
    });

    view.querySelector('#btn-clear-search').addEventListener('click', () => {
        // Clear all search inputs
        view.querySelector('#search-phone').value = '';
        view.querySelector('#search-email').value = '';
        view.querySelector('#search-identity').value = '';
        view.querySelector('#search-account').value = '';
        // Reload all customers
        loadAllCustomers();
    });

    // Allow Enter key to trigger search
    ['#search-phone', '#search-email', '#search-identity', '#search-account'].forEach((inputId, index) => {
        const searchTypes = ['phone', 'email', 'identity', 'account'];
        view.querySelector(inputId).addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const query = e.target.value;
                performSearch(query, searchTypes[index]);
            }
        });
    });

    view.addEventListener('click', (e) => {
        const viewButton = e.target.closest('.view-customer-btn');
        if (viewButton) {
            showCustomerDetail(viewButton.dataset.id, loadAllCustomers);
        }
        if (e.target.closest('#new-customer-btn')) {
            showNewCustomerModal(loadAllCustomers);
        }
    });

    return view;
}

