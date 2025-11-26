import { Folder, File, Upload, Download, Trash2, Plus, RefreshCw, HardDrive, CheckCircle, XCircle, AlertCircle } from 'lucide-static';
import { hdfsHealth, hdfsList, hdfsFileStatus, hdfsCreateDirectory, hdfsDeleteFile, hdfsDirectorySize, hdfsUploadFile, hdfsDownloadFile } from '../services/api.js';
import { formatDate, formatBytes } from '../utils/helpers.js';
import { animateStaggerIn } from '../utils/animations.js';

function showMessage(element, message, isSuccess = true) {
    element.textContent = message;
    element.className = `text-center text-sm mt-2 h-auto ${isSuccess ? 'text-success' : 'text-danger'}`;
    setTimeout(() => {
        element.textContent = '';
        element.className = 'text-center text-sm mt-2 h-4';
    }, 5000);
}

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    return formatBytes(bytes);
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    try {
        return new Date(timestamp).toLocaleString();
    } catch {
        return timestamp;
    }
}

export function HDFS() {
    const view = document.createElement('div');
    let currentPath = '/banktrading';
    let healthStatus = null;

    const renderView = async () => {
        try {
            // Check HDFS health
            healthStatus = await hdfsHealth();
        } catch (error) {
            console.error('HDFS health check failed:', error);
            healthStatus = { status: 'unhealthy', error: error.message };
        }

        view.innerHTML = `
            <div class="space-y-6">
                <!-- Header -->
                <div class="flex justify-between items-center">
                    <div>
                        <h2 class="text-2xl font-bold text-secondary-dark flex items-center gap-2">
                            ${HardDrive} HDFS Storage Management
                        </h2>
                        <p class="text-secondary mt-1">Quản lý lưu trữ dữ liệu trên HDFS</p>
                    </div>
                    <div class="flex items-center gap-2">
                        <button id="refresh-btn" class="px-4 py-2 bg-primary text-background rounded-lg hover:bg-primary-dark transition-colors flex items-center gap-2">
                            ${RefreshCw} Làm mới
                        </button>
                    </div>
                </div>

                <!-- Health Status -->
                <div id="health-status" class="bg-surface rounded-xl shadow-md p-4">
                    ${healthStatus?.status === 'healthy' 
                        ? `<div class="flex items-center gap-2 text-success">${CheckCircle} HDFS đang hoạt động bình thường</div>`
                        : `<div class="flex items-center gap-2 text-danger">${XCircle} HDFS không khả dụng: ${healthStatus?.error || 'Unknown error'}</div>`
                    }
                </div>

                <!-- Path Navigation -->
                <div class="bg-surface rounded-xl shadow-md p-4">
                    <div class="flex items-center gap-2 mb-4">
                        <label class="text-sm font-medium text-secondary-dark">Đường dẫn:</label>
                        <input 
                            type="text" 
                            id="path-input" 
                            value="${currentPath}" 
                            class="flex-1 bg-background border-border rounded-lg p-2 focus:outline-none focus:ring-2 focus:ring-primary"
                            placeholder="/banktrading"
                        />
                        <button id="navigate-btn" class="px-4 py-2 bg-primary text-background rounded-lg hover:bg-primary-dark transition-colors">
                            Điều hướng
                        </button>
                    </div>
                    <div class="flex gap-2">
                        <button id="create-dir-btn" class="px-4 py-2 bg-secondary-light text-secondary-dark rounded-lg hover:bg-border transition-colors flex items-center gap-2">
                            ${Plus} Tạo thư mục
                        </button>
                        <label for="file-upload-input" class="px-4 py-2 bg-primary text-background rounded-lg hover:bg-primary-dark transition-colors flex items-center gap-2 cursor-pointer">
                            ${Upload} Tải lên
                        </label>
                        <input type="file" id="file-upload-input" class="hidden" multiple />
                        <button id="back-btn" class="px-4 py-2 bg-secondary-light text-secondary-dark rounded-lg hover:bg-border transition-colors">
                            ← Quay lại
                        </button>
                    </div>
                </div>

                <!-- Directory Size Info -->
                <div id="size-info" class="bg-surface rounded-xl shadow-md p-4">
                    <div class="text-sm text-secondary">Đang tải thông tin...</div>
                </div>

                <!-- File List -->
                <div class="bg-surface rounded-xl shadow-md overflow-hidden">
                    <div class="p-4 border-b border-border">
                        <h3 class="text-lg font-semibold text-secondary-dark">Nội dung thư mục</h3>
                    </div>
                    <div id="file-list" class="p-4">
                        <div class="text-center text-secondary">Đang tải...</div>
                    </div>
                </div>

                <!-- Create Directory Modal -->
                <div id="create-dir-modal" class="hidden fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div class="bg-surface rounded-xl p-6 max-w-md w-full mx-4">
                        <h3 class="text-lg font-semibold text-secondary-dark mb-4">Tạo thư mục mới</h3>
                        <input 
                            type="text" 
                            id="new-dir-name" 
                            placeholder="Tên thư mục"
                            class="w-full bg-background border-border rounded-lg p-2 mb-4 focus:outline-none focus:ring-2 focus:ring-primary"
                        />
                        <div id="create-dir-response" class="text-sm mb-4 h-4"></div>
                        <div class="flex gap-2">
                            <button id="create-dir-submit" class="flex-1 px-4 py-2 bg-primary text-background rounded-lg hover:bg-primary-dark transition-colors">
                                Tạo
                            </button>
                            <button id="create-dir-cancel" class="flex-1 px-4 py-2 bg-secondary-light text-secondary-dark rounded-lg hover:bg-border transition-colors">
                                Hủy
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Load initial data
        await loadDirectorySize();
        await loadFileList();

        // Event listeners
        setupEventListeners();

        return view;
    };

    const setupEventListeners = () => {
        // Refresh button
        const refreshBtn = view.querySelector('#refresh-btn');
        refreshBtn?.addEventListener('click', async () => {
            await loadDirectorySize();
            await loadFileList();
        });

        // Navigate button
        const navigateBtn = view.querySelector('#navigate-btn');
        navigateBtn?.addEventListener('click', async () => {
            const pathInput = view.querySelector('#path-input');
            currentPath = pathInput.value || '/banktrading';
            await loadFileList();
        });

        // Back button
        const backBtn = view.querySelector('#back-btn');
        backBtn?.addEventListener('click', () => {
            if (currentPath !== '/') {
                const parts = currentPath.split('/').filter(p => p);
                parts.pop();
                currentPath = parts.length > 0 ? '/' + parts.join('/') : '/banktrading';
                view.querySelector('#path-input').value = currentPath;
                loadFileList();
            }
        });

        // Create directory button
        const createDirBtn = view.querySelector('#create-dir-btn');
        createDirBtn?.addEventListener('click', () => {
            const modal = view.querySelector('#create-dir-modal');
            modal.classList.remove('hidden');
        });

        // Create directory submit
        const createDirSubmit = view.querySelector('#create-dir-submit');
        createDirSubmit?.addEventListener('click', async () => {
            const dirName = view.querySelector('#new-dir-name').value.trim();
            const responseEl = view.querySelector('#create-dir-response');
            
            if (!dirName) {
                showMessage(responseEl, 'Vui lòng nhập tên thư mục', false);
                return;
            }

            try {
                const newPath = currentPath.endsWith('/') 
                    ? `${currentPath}${dirName}` 
                    : `${currentPath}/${dirName}`;
                await hdfsCreateDirectory(newPath);
                showMessage(responseEl, 'Tạo thư mục thành công!', true);
                view.querySelector('#new-dir-name').value = '';
                view.querySelector('#create-dir-modal').classList.add('hidden');
                await loadFileList();
            } catch (error) {
                showMessage(responseEl, `Lỗi: ${error.message}`, false);
            }
        });

        // Create directory cancel
        const createDirCancel = view.querySelector('#create-dir-cancel');
        createDirCancel?.addEventListener('click', () => {
            view.querySelector('#create-dir-modal').classList.add('hidden');
            view.querySelector('#new-dir-name').value = '';
        });

        // File upload
        const fileUploadInput = view.querySelector('#file-upload-input');
        fileUploadInput?.addEventListener('change', async (e) => {
            const files = Array.from(e.target.files);
            if (files.length === 0) return;

            const fileList = view.querySelector('#file-list');
            const originalContent = fileList.innerHTML;
            fileList.innerHTML = '<div class="text-center text-secondary py-4">Đang tải lên...</div>';

            try {
                for (const file of files) {
                    const hdfsPath = currentPath.endsWith('/') 
                        ? `${currentPath}${file.name}` 
                        : `${currentPath}/${file.name}`;
                    await hdfsUploadFile(file, hdfsPath);
                }
                showMessage(fileList, `Đã tải lên ${files.length} tệp thành công!`, true);
                await loadFileList();
                await loadDirectorySize();
            } catch (error) {
                fileList.innerHTML = originalContent;
                showMessage(fileList, `Lỗi: ${error.message}`, false);
            } finally {
                e.target.value = ''; // Reset input
            }
        });
    };

    const loadDirectorySize = async () => {
        const sizeInfo = view.querySelector('#size-info');
        try {
            const sizeData = await hdfsDirectorySize(currentPath);
            sizeInfo.innerHTML = `
                <div class="grid grid-cols-3 gap-4 text-sm">
                    <div>
                        <div class="text-secondary">Thư mục</div>
                        <div class="text-lg font-semibold text-secondary-dark">${sizeData.directoryCount || 0}</div>
                    </div>
                    <div>
                        <div class="text-secondary">Tệp tin</div>
                        <div class="text-lg font-semibold text-secondary-dark">${sizeData.fileCount || 0}</div>
                    </div>
                    <div>
                        <div class="text-secondary">Kích thước</div>
                        <div class="text-lg font-semibold text-secondary-dark">${formatFileSize(sizeData.length || 0)}</div>
                    </div>
                </div>
            `;
        } catch (error) {
            sizeInfo.innerHTML = `<div class="text-sm text-danger">Lỗi: ${error.message}</div>`;
        }
    };

    const loadFileList = async () => {
        const fileList = view.querySelector('#file-list');
        fileList.innerHTML = '<div class="text-center text-secondary">Đang tải...</div>';

        try {
            const data = await hdfsList(currentPath);
            
            if (!data.items || data.items.length === 0) {
                fileList.innerHTML = '<div class="text-center text-secondary py-8">Thư mục trống</div>';
                return;
            }

            // Sort: directories first, then files
            const sortedItems = [...data.items].sort((a, b) => {
                if (a.type === 'DIRECTORY' && b.type !== 'DIRECTORY') return -1;
                if (a.type !== 'DIRECTORY' && b.type === 'DIRECTORY') return 1;
                return a.path.localeCompare(b.path);
            });

            fileList.innerHTML = `
                <table class="w-full text-sm">
                    <thead class="bg-secondary-light text-secondary-dark uppercase text-xs">
                        <tr>
                            <th class="text-left p-3">Tên</th>
                            <th class="text-left p-3">Loại</th>
                            <th class="text-right p-3">Kích thước</th>
                            <th class="text-left p-3">Ngày sửa đổi</th>
                            <th class="text-center p-3">Thao tác</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${sortedItems.map(item => `
                            <tr class="border-b border-border hover:bg-secondary-light/50">
                                <td class="p-3">
                                    <div class="flex items-center gap-2">
                                        ${item.type === 'DIRECTORY' ? Folder : File}
                                        <span class="font-medium text-secondary-dark cursor-pointer hover:text-primary" 
                                              data-path="${currentPath.endsWith('/') ? currentPath + item.path : currentPath + '/' + item.path}">
                                            ${item.path || '..'}
                                        </span>
                                    </div>
                                </td>
                                <td class="p-3">
                                    <span class="px-2 py-1 rounded text-xs ${item.type === 'DIRECTORY' ? 'bg-blue-900/50 text-blue-300' : 'bg-gray-700 text-gray-300'}">
                                        ${item.type === 'DIRECTORY' ? 'Thư mục' : 'Tệp tin'}
                                    </span>
                                </td>
                                <td class="p-3 text-right">${formatFileSize(item.length)}</td>
                                <td class="p-3">${formatTimestamp(item.modificationTime)}</td>
                                <td class="p-3 text-center">
                                    ${item.type === 'DIRECTORY' 
                                        ? `<button class="p-1 text-primary hover:text-primary-dark" data-action="navigate" data-path="${currentPath.endsWith('/') ? currentPath + item.path : currentPath + '/' + item.path}">
                                            ${Folder} Mở
                                          </button>`
                                        : `<button class="p-1 text-success hover:text-success-dark" 
                                            data-action="download" 
                                            data-path="${currentPath.endsWith('/') ? currentPath + item.path : currentPath + '/' + item.path}"
                                            title="Tải xuống">
                                            ${Download}
                                          </button>`
                                    }
                                    <button class="p-1 text-danger hover:text-danger-dark ml-2" 
                                            data-action="delete" 
                                            data-path="${currentPath.endsWith('/') ? currentPath + item.path : currentPath + '/' + item.path}"
                                            data-type="${item.type}"
                                            title="Xóa">
                                        ${Trash2}
                                    </button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;

            // Add click handlers
            fileList.querySelectorAll('[data-action="navigate"]').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const path = e.currentTarget.dataset.path;
                    currentPath = path;
                    view.querySelector('#path-input').value = currentPath;
                    await loadFileList();
                    await loadDirectorySize();
                });
            });

            fileList.querySelectorAll('[data-action="download"]').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const path = e.currentTarget.dataset.path;
                    try {
                        await hdfsDownloadFile(path);
                        showMessage(view.querySelector('#file-list'), 'Tải xuống thành công!', true);
                    } catch (error) {
                        showMessage(view.querySelector('#file-list'), `Lỗi: ${error.message}`, false);
                    }
                });
            });

            fileList.querySelectorAll('[data-action="delete"]').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const path = e.currentTarget.dataset.path;
                    const type = e.currentTarget.dataset.type;
                    const confirmMessage = type === 'DIRECTORY' 
                        ? `Bạn có chắc muốn xóa thư mục "${path}"?`
                        : `Bạn có chắc muốn xóa tệp "${path}"?`;
                    
                    if (confirm(confirmMessage)) {
                        try {
                            await hdfsDeleteFile(path, type === 'DIRECTORY');
                            showMessage(view.querySelector('#file-list'), 'Xóa thành công!', true);
                            await loadFileList();
                            await loadDirectorySize();
                        } catch (error) {
                            showMessage(view.querySelector('#file-list'), `Lỗi: ${error.message}`, false);
                        }
                    }
                });
            });

            // Click on file/directory name to navigate
            fileList.querySelectorAll('[data-path]').forEach(el => {
                if (el.closest('td').querySelector('[data-action="navigate"]')) {
                    el.style.cursor = 'pointer';
                    el.addEventListener('click', (e) => {
                        const path = e.currentTarget.dataset.path;
                        currentPath = path;
                        view.querySelector('#path-input').value = currentPath;
                        loadFileList();
                        loadDirectorySize();
                    });
                }
            });

        } catch (error) {
            fileList.innerHTML = `<div class="text-center text-danger py-8">Lỗi: ${error.message}</div>`;
        }
    };

    // Initialize
    renderView();

    return view;
}

