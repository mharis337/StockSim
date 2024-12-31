"use client";

import ModelTraining from "@/components/ModelTraining";
import ModelManager from "@/components/ModelManager";

export default function ModelTrainingPage() {
  return (
    <div className="min-h-screen p-8 bg-gradient-to-b from-blue-50 to-blue-100">
      <div className="max-w-[1400px] mx-auto">
        <ModelTraining />
      </div>
      <div className="max-w-[1400px] mx-auto">
        <ModelManager />
      </div>
    </div>
  );
}