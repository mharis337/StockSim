export const getToken = () => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('token');
    }
    return null;
  };
  
  export const isAuthenticated = () => !!getToken();
  
  export const getAuthHeaders = () => ({
    'Authorization': `Bearer ${getToken()}`,
    'Content-Type': 'application/json'
  });