import { X } from 'lucide-static';
import { animateModalOpen, animateModalClose } from '../utils/animations.js';

/**
 * Creates and manages a modal dialog.
 * @param {object} options
 * @param {string} options.title - The title of the modal.
 * @param {HTMLElement | string} options.content - The content to display in the modal.
 * @param {string} [options.size='max-w-2xl'] - Tailwind CSS class for modal width.
 * @param {Function} [options.onClose] - Callback function when modal is closed.
 */
export function Modal({ title, content, size = 'max-w-2xl', onClose }) {
    const modalOverlay = document.createElement('div');
    modalOverlay.className = 'fixed inset-0 bg-black bg-opacity-70 z-40 flex items-center justify-center p-4';
    modalOverlay.style.opacity = 0;

    const modalEl = document.createElement('div');
    modalEl.className = `bg-surface rounded-xl shadow-md w-full ${size} flex flex-col`;
    
    modalEl.innerHTML = `
        <div class="flex items-center justify-between p-4 border-b border-border">
            <h3 class="text-lg font-semibold text-secondary-dark">${title}</h3>
            <button id="modal-close-btn" class="text-secondary hover:text-primary">${X}</button>
        </div>
        <div class="p-6 overflow-y-auto"></div>
    `;

    const contentContainer = modalEl.querySelector('.p-6');
    if (typeof content === 'string') {
        contentContainer.innerHTML = content;
    }
    else {
        contentContainer.appendChild(content);
    }

    modalOverlay.appendChild(modalEl);
    document.body.appendChild(modalOverlay);
    document.body.style.overflow = 'hidden';

    animateModalOpen(modalOverlay, modalEl);

    const closeModal = () => {
        animateModalClose(modalOverlay, modalEl, () => {
            if (modalOverlay.parentNode) {
                document.body.removeChild(modalOverlay);
            }
            document.body.style.overflow = '';
            if (onClose) onClose();
        });
    };

    modalOverlay.querySelector('#modal-close-btn').addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            closeModal();
        }
    });

    return {
        close: closeModal,
    };
}
