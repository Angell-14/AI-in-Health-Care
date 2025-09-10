"use client";

import { motion } from "framer-motion";
import { FloatingDock } from "../ui/floating-dock";
import React from "react";
import { useRouter } from "next/navigation";
import { AuroraBackground } from "../ui/aurora-background";
import { CardHoverEffectDemo, MainSection } from "./mainsection";



export function AuroraBackgroundDemo() {
    const router = useRouter(); 
    return (
    (<AuroraBackground>
      <motion.div
        initial={{ opacity: 0.0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{
          delay: 0.3,
          duration: 0.8,
          ease: "easeInOut",
        }}
        className="relative flex flex-col gap-4 items-center justify-center px-4">
        <div className="text-3xl md:text-7xl font-bold dark:text-white text-center">
          Smarter reports
        </div>
        <div
          className="font-extralight text-base md:text-4xl font-semibold dark:text-neutral-200 py-4">
          faster care
        </div>
        <CardHoverEffectDemo/>
        {/* <FloatingDock items={projects} /> */}
        <button
          className="bg-black dark:bg-white rounded-full w-fit text-white dark:text-black px-4 py-2"
          onClick={() => {
            router.push("/#mainsection");
          }}>
          check now !
        </button>
      </motion.div>
    </AuroraBackground>)
  );
}
