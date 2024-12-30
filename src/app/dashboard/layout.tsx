"use client";

import ProtectedLayout from "@/components/ProtectedLayout";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProtectedLayout currentPage="dashboard">
      {children}
    </ProtectedLayout>
  );
}