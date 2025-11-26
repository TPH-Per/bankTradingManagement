export function Settings() {
    const view = document.createElement('div');
    view.innerHTML = `
        <h2 class="text-2xl font-bold text-secondary-dark mb-6">Cài đặt</h2>
        <div class="bg-surface rounded-xl shadow-subtle p-8 text-center">
            <h3 class="text-lg font-semibold text-secondary-dark">Cài đặt hệ thống & Kiểm toán</h3>
            <p class="text-secondary mt-2">Phần này đang được xây dựng. Cấu hình hệ thống và nhật ký kiểm toán sẽ được quản lý tại đây.</p>
        </div>
    `;
    return view;
}
