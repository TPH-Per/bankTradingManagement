const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

async function request(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  try {
    const response = await fetch(url, { ...options, headers });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ message: `Request failed with status ${response.status}` }));
      throw new Error(errorData.detail || errorData.message);
    }
    return response.status === 204 ? null : response.json();
  } catch (error) {
    console.error(`API Error on ${options.method || 'GET'} ${endpoint}:`, error);
    if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        // This is a network-level error. The server was unreachable.
        throw new Error(`Connection Failed: The application could not reach the API server. This is usually due to the backend server being offline or a network issue. Please check that the server at '${BASE_URL}' is running and accessible.`);
    }
    throw error;
  }
}

// Customer Management
export const getCustomers = (params) => request(`/customers?${new URLSearchParams(params)}`);
export const getCustomerById = (accountId) => request(`/customers/${accountId}`);
export const createCustomer = (data) => request('/customers', { method: 'POST', body: JSON.stringify(data) });
export const updateCustomer = (accountId, data) => request(`/customers/${accountId}`, { method: 'PATCH', body: JSON.stringify(data) });
export const searchCustomers = (query) => request(`/customers/search?${new URLSearchParams(query)}`);

// Transaction Management
export const createTransaction = (data) => request('/rt/transactions', { method: 'POST', body: JSON.stringify(data) });
export const getTransactions = (params) => request(`/rt/transactions?${new URLSearchParams(params)}`);
export const getAllTransactions = (params) => request(`/rt/transactions/all?${new URLSearchParams(params)}`);

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
