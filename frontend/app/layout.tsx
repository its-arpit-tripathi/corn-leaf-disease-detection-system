import type { Metadata } from "next";
import "@fontsource-variable/bricolage-grotesque";
import "@fontsource-variable/instrument-sans";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import Header from "@/components/Header";

export const metadata: Metadata = {
  title: "Leafcheck: corn leaf disease detection",
  description: "Upload a photo of a corn leaf and see the likely disease, the confidence, and where the model looked.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <AuthProvider>
          <Header />
          <main className="mx-auto max-w-6xl px-4 py-8 sm:py-12">{children}</main>
        </AuthProvider>
      </body>
    </html>
  );
}
