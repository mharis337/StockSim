export interface LoginCredentials {
    email: string;
    password: string;
  }
  
  export interface AuthResponse {
    access_token: string;
    token_type: string;
  }
  
  export interface User {
    email: string;
    // Add other user fields
  }