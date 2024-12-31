"use client";

import ProtectedLayout from "@/components/ProtectedLayout";

export default function ModelTrainingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex-1 w-full">
        <ProtectedLayout currentPage="model-training">
          {children}
        </ProtectedLayout>
      </div>
    </div>
  );
}