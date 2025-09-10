import { AuroraBackground } from "@/components/ui/aurora-background";
import Image from "next/image";
import { motion } from "motion/react";
import { AuroraBackgroundDemo } from "@/components/pages/hero";
import { CardHoverEffectDemo } from "@/components/pages/mainsection";
import { FeaturesSectionDemo } from "@/components/ui/bento-grid";
import { FloatingDock } from "@/components/ui/floating-dock";

export const projects = [
  {
    title: "Chatbot",
    description: "AI-powered chatbot for patient engagement and symptom assessment.",
    href: "/chat/ch",
    icon: "🤖"
  },
  {
    title: "Personalize Treatment Plans",
    description: "Customized treatment recommendations based on patient data.",
    href: "/chat/p",
    icon: "📋"
  },
  {
    title: "Generate Data",
    description: "Customized and synthesized data",
    href: "/chat/d",
    icon: "📊"
  }
];
export default function Home() {
  return (
    <div>
      <AuroraBackgroundDemo/>
      <FeaturesSectionDemo/>
      <div style={{
        position: 'fixed',
        bottom: '20px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 50
      }}>
        <FloatingDock items={projects} />
      </div>
    </div>
  );
}
