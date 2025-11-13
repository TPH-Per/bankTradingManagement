import { Plus, Save, User, Hash, Mail, Calendar, ShieldCheck, BarChart, ArrowLeftRight } from 'lucide-static';
import { getCustomers, getCustomerById, getTransactions, getCustomerStats, updateCustomer, createCustomer } from '../services/api.js';
import { formatDate, formatCurrency } from '../utils/helpers.js';
import { Modal } from './Modal.js';
import { animateStaggerIn, showResponseMessage } from '../utils/animations.js';

function CustomerRow(customer) {
    const statusMap = {
        'active': 'bg-green-900/50 text-success',
        'inactive': 'bg-gray-700 text-gray-400',
        'suspended': 'bg-yellow-900/50 text-warning',
    };
    const statusClass = statusMap[customer.status] || statusMap['inactive'];

    return `
        <tr class="customer-row border-b border-border hover:bg-secondary-light">
            <td class="p-4">${customer.account_id}</td>
            <td class="p-4">
                <div class="font-medium text-secondary-dark">${customer.full_name}</div>
                <div class="text-sm text-secondary">${customer.email}</div>
            </td>
            <td class="p-4">${customer.national_id}</td>
            <td class="p-4">${formatDate(customer.created_at)}</td>
            <td class="p-4">
                <span class="px-2 py-1 text-xs font-semibold rounded-full ${statusClass}">
                    ${customer.status.charAt(0).toUpperCase() + customer.status.slice(1)}
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
            <td class="p-4 text-right"><div class="h-4 w-12 skeleton-loader inline-block"></div></td>
        </tr>
    `;
}

function createDetailModalContent() {
    const content = document.createElement('div');
    content.innerHTML = `
        <div class="flex border-b border-border">
            <button data-tab="profile" class="detail-tab-btn active p-3 font-medium flex items-center gap-2">${User} Hồ sơ</button>
            <button data-tab="transactions" class="detail-tab-btn p-3 font-medium flex items-center gap-2">${ArrowLeftRight} Giao dịch</button>
            <button data-tab="statistics" class="detail-tab-btn p-3 font-medium flex items-center gap-2">${BarChart} Thống kê</button>
        </div>
        <div class="mt-4 min-h-[300px]">
            <div id="profile-tab" class="detail-tab-content"></div>
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
                if (btn.dataset.tab === 'transactions' && !tabContainer.hasChildNodes()) {
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
        <div><label class="block text-sm font-medium text-secondary-dark mb-1">CMND/CCCD</label><input type="text" name="national_id" required class="w-full bg-background border-border rounded-lg p-2"></div>
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
            await createCustomer(data);
            showResponseMessage(responseEl, 'Tạo khách hàng thành công!', true);
            setTimeout(() => {
                modal.close();
                onUpdate();
            }, 1000);
        } catch (error) {
            showResponseMessage(responseEl, `Lỗi: ${error.message}`, false);
        }
    });
}

export function Customers() {
    const view = document.createElement('div');
    view.innerHTML = `
        <style>
            .detail-tab-btn.active { color: hsl(160, 80%, 45%); border-bottom: 2px solid hsl(160, 80%, 45%); }
        </style>
        <div class="flex justify-between items-center mb-6">
            <h2 class="text-2xl font-bold text-secondary-dark">Quản lý khách hàng</h2>
            <button id="new-customer-btn" class="bg-primary hover:bg-primary-dark text-background font-bold py-2 px-4 rounded-lg flex items-center gap-2">
                ${Plus}
                <span>Khách hàng mới</span>
            </button>
        </div>
        <div class="bg-surface rounded-xl shadow-md overflow-hidden">
            <div class="p-4">
                 <input type="text" placeholder="Tìm kiếm khách hàng theo tên, email, hoặc ID..." class="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary">
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-sm text-left">
                    <thead class="bg-secondary-light text-secondary-dark uppercase text-xs">
                        <tr>
                            <th class="p-4">ID Tài khoản</th><th class="p-4">Khách hàng</th><th class="p-4">CMND/CCCD</th>
                            <th class="p-4">Ngày tham gia</th><th class="p-4">Trạng thái</th><th class="p-4 text-right">Hành động</th>
                        </tr>
                    </thead>
                    <tbody id="customer-table-body"></tbody>
                </table>
            </div>
        </div>
    `;

    const tableBody = view.querySelector('#customer-table-body');
    
    async function loadCustomers() {
        tableBody.innerHTML = Array(5).fill(0).map(SkeletonRow).join('');
        try {
            const { items } = await getCustomers();
            tableBody.innerHTML = items.map(CustomerRow).join('');
            animateStaggerIn(view.querySelectorAll('.customer-row'));
        } catch (error) {
            tableBody.innerHTML = `<tr><td colspan="6" class="text-center p-8 text-danger"><p class="font-bold">Không thể tải khách hàng</p><p class="text-xs text-secondary mt-2 max-w-md mx-auto">${error.message}</p></td></tr>`;
        }
    }
    
    loadCustomers();

    view.addEventListener('click', (e) => {
        const viewButton = e.target.closest('.view-customer-btn');
        if (viewButton) {
            showCustomerDetail(viewButton.dataset.id, loadCustomers);
        }
        if (e.target.closest('#new-customer-btn')) {
            showNewCustomerModal(loadCustomers);
        }
    });

    return view;
}
