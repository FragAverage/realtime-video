import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FLUX.2 Klein Realtime - Webcam Stylization",
  description:
    "Real-time webcam stylization powered by FLUX.2 Klein 4B.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClerkProvider
      appearance={{
        variables: {
          colorPrimary: "#ffffff",
          colorBackground: "#18181b",
          colorText: "#fafafa",
          colorTextSecondary: "#a1a1aa",
          colorInputBackground: "#0a0a0a",
          colorInputText: "#fafafa",
          borderRadius: "0.75rem",
        },
        elements: {
          card: "bg-[#18181b] border border-white/10",
          formButtonPrimary: "bg-white text-black hover:bg-zinc-200",
          footerActionLink: "text-white hover:text-zinc-300",
        },
      }}
    >
      <html lang="en" className="dark">
        <body
          className={`${inter.variable} ${jetbrainsMono.variable} antialiased min-h-screen`}
        >
          {children}
        </body>
      </html>
    </ClerkProvider>
  );
}
