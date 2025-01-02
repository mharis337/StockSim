import React from 'react';
import { Play, Trash2 } from 'lucide-react';

interface Model {
  id: string;
  _id?: string;
  name: string;
  uploadDate: string;
}

interface ModelListProps {
  models: Model[];
  selectedModel: Model | null;
  onSelectModel: (model: Model) => void;
  onDeleteModel: (model: Model) => Promise<void>;
}

export const ModelList: React.FC<ModelListProps> = ({
  models,
  selectedModel,
  onSelectModel,
  onDeleteModel,
}) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
    {models.map((model) => (
      <div
        key={model.id}
        className={`p-4 rounded-lg border ${
          selectedModel?.id === model.id
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-200'
        }`}
      >
        <div className="flex justify-between items-start">
          <div>
            <h4 className="font-medium">{model.name}</h4>
            <p className="text-sm text-gray-600">
              Uploaded: {new Date(model.uploadDate).toLocaleDateString()}
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => onSelectModel(model)}
              className={`p-2 rounded ${
                selectedModel?.id === model.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
              aria-label={`Select model ${model.name}`}
            >
              <Play className="w-4 h-4" />
            </button>
            <button
              onClick={() => onDeleteModel(model)}
              className="p-2 bg-gray-100 text-gray-600 rounded hover:bg-gray-200"
              aria-label={`Delete model ${model.name}`}
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    ))}
  </div>
);
