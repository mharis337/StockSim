"use client";

import ProtectedLayout from "@/components/ProtectedLayout";

export default function TradeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProtectedLayout currentPage="trade">
      {children}
    </ProtectedLayout>
  );
}