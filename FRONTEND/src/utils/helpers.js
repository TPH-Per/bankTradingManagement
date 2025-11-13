/**
 * Formats a number as a currency string.
 * @param {number} amount The amount to format.
 * @param {string} currency The currency code (e.g., 'USD', 'VND').
 * @returns {string} The formatted currency string.
 */
export function formatCurrency(amount, currency = 'VND') {
  return new Intl.NumberFormat('vi-VN', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 0, // VND typically doesn't use decimal places
  }).format(amount);
}

/**
 * Formats a date string or Date object into a more readable format.
 * @param {string | Date} date The date to format.
 * @returns {string} The formatted date string (e.g., 'Apr 25, 2025').
 */
export function formatDate(date) {
  if (!date) return 'N/A';
  return new Date(date).toLocaleDateString('vi-VN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

/**
 * Debounces a function to limit the rate at which it gets called.
 * @param {Function} func The function to debounce.
 * @param {number} delay The debounce delay in milliseconds.
 * @returns {Function} The debounced function.
 */
export function debounce(func, delay) {
  let timeoutId;
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
}
