import React, { useState, useEffect, useCallback } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, Play, Pause, Trash2, AlertCircle } from 'lucide-react';

interface Model {
  _id: string;
  name: string;
  uploadDate: string;
  features?: string[];
  status: 'active' | 'inactive';
}

const ModelManager = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);

  const fetchModels = useCallback(async () => {
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');

      const response = await fetch('http://localhost:5000/api/models', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) throw new Error('Failed to fetch models');
      const data = await response.json();
      console.log('Fetched models:', data.models); // Debug log
      setModels(data.models || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch models');
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleModelUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validExtensions = ['.h5', '.keras', '.pkl', '.joblib'];
    const fileExtension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!validExtensions.includes(fileExtension)) {
      setError(`Invalid file type. Supported formats: ${validExtensions.join(', ')}`);
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('model', file);

    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');

      const response = await fetch('http://localhost:5000/api/models/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload model');
      }

      await fetchModels();
      setUploadProgress(100);
      setTimeout(() => setUploadProgress(0), 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload model');
    } finally {
      setIsUploading(false);
    }
  };

  const toggleModelStatus = async (modelId: string) => {
    console.log('Toggling model with ID:', modelId); // Debug log
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');

      const response = await fetch(`http://localhost:5000/api/models/${modelId}/toggle`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to toggle model status');
      }
      
      const data = await response.json();
      console.log('Toggle response:', data); // Debug log
      setActiveModel(data.status === 'active' ? modelId : null);
      await fetchModels();
    } catch (err) {
      console.error('Toggle error:', err); // Debug log
      setError(err instanceof Error ? err.message : 'Failed to toggle model');
    }
  };

  const deleteModel = async (modelId: string) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Authentication token not found');

      const response = await fetch(`http://localhost:5000/api/models/${modelId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to delete model');
      }
      
      if (activeModel === modelId) {
        setActiveModel(null);
      }
      await fetchModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete model');
    }
  };

  return (
    <div className="space-y-6 p-6 bg-white rounded-lg shadow">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-800">AI Model Manager</h2>
        <label className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors">
          <Upload className="w-5 h-5" />
          Upload Model
          <input
            type="file"
            accept=".h5,.keras,.pkl,.joblib"
            className="hidden"
            onChange={handleModelUpload}
            disabled={isUploading}
          />
        </label>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {isUploading && (
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}

      <div className="space-y-4">
        {models.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No models uploaded yet
          </div>
        ) : (
          models.map((model) => (
            <div 
              key={model._id}
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <h3 className="text-lg font-semibold text-gray-800">{model.name}</h3>
                  <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                    activeModel === model._id 
                      ? 'bg-green-100 text-green-700'
                      : 'bg-gray-100 text-gray-600'
                  }`}>
                    {activeModel === model._id ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <p className="text-sm text-gray-600">
                  Uploaded: {new Date(model.uploadDate).toLocaleDateString()}
                </p>
                {model.features && model.features.length > 0 && (
                  <p className="text-sm text-gray-600">
                    Features: {model.features.join(', ')}
                  </p>
                )}
                {activeModel === model._id && (
                  <div className="mt-2 p-2 bg-green-50 rounded-md">
                    <p className="text-sm text-green-700">
                      This model is currently active and processing market data in real-time.
                    </p>
                  </div>
                )}
              </div>
              
              <div className="flex items-center gap-4">
                <button
                  onClick={() => toggleModelStatus(model._id)}
                  className={`p-2 rounded-lg transition-colors ${
                    activeModel === model._id
                      ? 'bg-red-100 text-red-600 hover:bg-red-200'
                      : 'bg-green-100 text-green-600 hover:bg-green-200'
                  }`}
                  title={activeModel === model._id ? 'Deactivate Model' : 'Activate Model'}
                >
                  {activeModel === model._id ? (
                    <Pause className="w-5 h-5" />
                  ) : (
                    <Play className="w-5 h-5" />
                  )}
                </button>
                
                <button
                  onClick={() => deleteModel(model._id)}
                  className="p-2 bg-gray-100 text-gray-600 rounded-lg hover:bg-gray-200 transition-colors"
                  title="Delete Model"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ModelManager;