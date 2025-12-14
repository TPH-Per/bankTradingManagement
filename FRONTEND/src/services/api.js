import cacheService from './cache.js';
import retryService from './retry.js';
import errorHandler from './errorHandler.js';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

/**
 * Enhanced request function with caching, retry, and error handling
 */
async function request(endpoint, options = {}) {
  const {
    useCache = false,
    cacheTTL = null,
    retry = true,
    showError = true,
    ...fetchOptions
  } = options;

  const url = `${BASE_URL}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  };

  // Check cache for GET requests
  if (useCache && fetchOptions.method === 'GET' || !fetchOptions.method) {
    const cached = cacheService.get(endpoint, fetchOptions.params || {});
    if (cached) {
      return cached;
    }
  }

  // Create request function for retry service
  const makeRequest = async () => {
    const response = await fetch(url, { ...fetchOptions, headers });

    if (!response.ok) {
      const error = new Error();
      error.response = response;
      error.status = response.status;
      try {
        const errorData = await response.json();
        error.message = errorData.detail || errorData.message || `Request failed with status ${response.status}`;
      } catch {
        error.message = `Request failed with status ${response.status}`;
      }
      throw error;
    }

    const data = response.status === 204 ? null : await response.json();

    // Cache successful GET responses
    if (useCache && (fetchOptions.method === 'GET' || !fetchOptions.method)) {
      cacheService.set(endpoint, fetchOptions.params || {}, data, cacheTTL);
    }

    return data;
  };

  try {
    // Use retry service if enabled
    const data = retry
      ? await retryService.retry(makeRequest, { maxRetries: retry === true ? 3 : retry })
      : await makeRequest();

    return data;
  } catch (error) {
    console.error(`API Error on ${fetchOptions.method || 'GET'} ${endpoint}:`, error);

    // Handle error through error handler
    if (showError) {
      errorHandler.handleError(error, { endpoint, method: fetchOptions.method || 'GET' });
    }

    throw error;
  }
}

// Account Management
export const createAccount = (data) => request('/accounts', { method: 'POST', body: JSON.stringify(data) });
export const getAccount = (accountId) => request(`/accounts/${accountId}`);
export const updateAccount = (accountId, data) => request(`/accounts/${accountId}`, { method: 'PATCH', body: JSON.stringify(data) });
export const listAccounts = (params) => request(`/accounts?${new URLSearchParams(params)}`);
export const getAllAccounts = () => request('/accounts/all');


// Customer Management (compatibility - maps to accounts)
export const getCustomers = (params) => request(`/customers?${new URLSearchParams(params)}`);
export const getCustomerById = (accountId) => request(`/customers/${accountId}`);
export const createCustomer = (data) => request('/customers', { method: 'POST', body: JSON.stringify(data) });
export const updateCustomer = (accountId, data) => request(`/customers/${accountId}`, { method: 'PATCH', body: JSON.stringify(data) });
export const searchCustomers = (query) => request(`/customers/search?${new URLSearchParams(query)}`);

// Transaction Management
export const createTransaction = (data) => request('/rt/transactions', { method: 'POST', body: JSON.stringify(data) });
export const getTransactions = (params) => request(`/rt/transactions?${new URLSearchParams(params)}`);
export const getAllTransactions = (params) => request(`/rt/transactions/all?${new URLSearchParams(params)}`);

// Transfer Management
export const getTransfers = (params) => request(`/rt/transfers?${new URLSearchParams(params)}`);

// Account validation for P2P transactions
export const validateAccount = (accountId) => request(`/accounts/${accountId}`);

// Account Balance (cached for 5 minutes)
export const getAccountBalance = (accountId) => request(`/accounts/${accountId}/balance`, {
  useCache: true,
  cacheTTL: 5 * 60 * 1000, // 5 minutes
});
export const updateAccountBalance = async (accountId, data) => {
  const result = await request(`/accounts/${accountId}/balance/update`, { method: 'POST', body: JSON.stringify(data) });
  // Invalidate cache for this account's balance after update
  cacheService.invalidate(`/accounts/${accountId}/balance`);
  return result;
};


// Account Statement (Sao kê)
export const getAccountStatement = (accountId, params) => request(`/accounts/${accountId}/statement?${new URLSearchParams(params)}`);
export const generateDailySnapshot = (accountId, day) => request(`/accounts/${accountId}/statement/generate?day=${day}`, { method: 'POST' });

// Dashboard (cached for 1 minute)
export const getDashboardStats = (params) => request(`/dashboard/stats?${new URLSearchParams(params)}`, {
  useCache: true,
  cacheTTL: 60 * 1000, // 1 minute
  params,
});

// Reports
export const getReportsStats = (params) => request(`/reports/stats?${new URLSearchParams(params)}`);

// Statistics
export const getCustomerStats = (accountId, params) => request(`/customers/${accountId}/stats/summary?${new URLSearchParams(params)}`);
export const getCompanyDailyStats = (date, params) => request(`/company/stats/daily/${date}?${new URLSearchParams(params)}`);
export const getCompanyMonthlyStats = (yearMonth, params) => request(`/company/stats/monthly/${yearMonth}?${new URLSearchParams(params)}`);

// ML Predictions
export const prepareFeatures = () => request('/ml/prepare-features');
export const predictCashIn = (data) => request('/ml/predict/cash-in', { method: 'POST', body: JSON.stringify(data) });
export const predictCashOut = (data) => request('/ml/predict/cash-out', { method: 'POST', body: JSON.stringify(data) });

// Health Check
export const getHealth = () => request('/health');

// HDFS Operations
export const hdfsHealth = () => request('/hdfs/health');
export const hdfsList = (path = '/banktrading') => request(`/hdfs/list?path=${encodeURIComponent(path)}`);
export const hdfsFileStatus = (path) => request(`/hdfs/status?path=${encodeURIComponent(path)}`);
export const hdfsCreateDirectory = (path) => request(`/hdfs/mkdir?path=${encodeURIComponent(path)}`, { method: 'POST' });

// Upload file (multipart/form-data)
export const hdfsUploadFile = async (file, hdfsPath) => {
  const formData = new FormData();
  formData.append('file', file);
  const url = `${BASE_URL}/hdfs/upload?hdfs_path=${encodeURIComponent(hdfsPath)}`;
  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: `Request failed with status ${response.status}` }));
    throw new Error(errorData.detail || errorData.message);
  }
  return response.json();
};

// Download file (returns blob)
export const hdfsDownloadFile = async (hdfsPath) => {
  const url = `${BASE_URL}/hdfs/download?hdfs_path=${encodeURIComponent(hdfsPath)}`;
  const response = await fetch(url);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ message: `Request failed with status ${response.status}` }));
    throw new Error(errorData.detail || errorData.message);
  }
  const blob = await response.blob();
  const filename = hdfsPath.split('/').pop() || 'download';
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = downloadUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(downloadUrl);
  return { status: 'success', filename };
};

export const hdfsDeleteFile = (path, recursive = false) => request(`/hdfs/delete?path=${encodeURIComponent(path)}&recursive=${recursive}`, { method: 'DELETE' });
export const hdfsDirectorySize = (path = '/banktrading') => request(`/hdfs/size?path=${encodeURIComponent(path)}`);