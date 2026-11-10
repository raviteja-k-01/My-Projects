import { ExternalLink, Github, BarChart2, Car, UserMinus, ShoppingCart } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LucideIcon } from "lucide-react";

type ProjectLinks = {
  project?: string;
  powerbi?: string;
  github: string;
};

const Projects = () => {
  const projects: {
    title: string;
    description: string;
    dataset: string;
    keyOutcome: string;
    tools: string[];
    gradient: string;
    icon: LucideIcon;
    links: ProjectLinks;
  }[] = [
    {
      title: "Uber Ride Cancellations & Wasted Capacity Analysis",
      description:
        "Analyzed large-scale ride data to identify cancellation drivers, peak-hour inefficiencies, and revenue-impacting demand–supply gaps for operational decision-making.",
      dataset: "150,000+ ride records (NCR region)",
      keyOutcome:
        "38% cancellation rate identified; ₹19.9M estimated revenue impact from wasted capacity",
      tools: ["SQL", "Python (pandas)", "Power BI"],
      gradient: "from-primary/20 to-secondary/20",
      icon: Car,
      links: {
        powerbi:
          "https://drive.google.com/file/d/14FZn1iow0VnutJfEX-TrLvFkcxymq2cK/view?usp=sharing",
        github: "https://github.com/raviteja-k-01",
      },
    },
    {
      title: "Customer Churn Analysis",
      description:
        "Analyzed customer behavior and contract data to identify churn drivers and segment customers by churn risk for retention-focused decision-making.",
      dataset: "7,043 customer records (21 features)",
      keyOutcome:
        "55–60% customers identified as high-risk; key churn drivers isolated for retention strategy",
      tools: ["Python (pandas)", "Matplotlib", "Seaborn"],
      gradient: "from-secondary/20 to-primary/20",
      icon: UserMinus,
      links: {
        github: "https://github.com/raviteja-k-01",
      },
    },
    {
      title: "Retail Sales Insights & Customer Segmentation",
      description:
        "Performed end-to-end retail sales analysis using RFM segmentation to uncover customer value patterns and support targeted marketing decisions.",
      dataset: "Online retail transactional records (cleaned & processed)",
      keyOutcome:
        "5 customer segments identified and depicted with an interactive Power BI dashboard",
      tools: ["Python (Pandas, NumPy)", "MySQL", "Power BI"],
      gradient: "from-primary/20 to-secondary/20",
      icon: ShoppingCart,
      links: {
        github:
          "https://github.com/raviteja-k-01/Retail-Sales-Insights-Customer-Segmentation",
      },
    },
  ];

  return (
    <section id="projects" className="py-20 relative overflow-hidden">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 animate-fade-in">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Featured Projects
            </h2>
            <div className="w-20 h-1 bg-gradient-to-r from-primary to-secondary mx-auto mb-8 rounded-full" />
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto">
              Real-world business analytics projects delivering actionable insights
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {projects.map((project, index) => (
              <div
                key={project.title}
                className="glass-card p-6 sm:p-8 hover:shadow-2xl transition-all duration-300 animate-slide-up group hover:scale-[1.02]"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div
                  className={`absolute inset-0 bg-gradient-to-br ${project.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg`}
                />

                <div className="relative z-10">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-primary to-secondary flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform">
                      <project.icon className="h-6 w-6 text-white" />
                    </div>
                    <h3 className="text-lg sm:text-xl font-bold text-foreground group-hover:text-primary transition-colors">
                      {project.title}
                    </h3>
                  </div>

                  <p className="text-sm sm:text-base text-muted-foreground mb-6">
                    {project.description}
                  </p>

                  <div className="space-y-3 mb-6">
                    <div className="flex items-start gap-2">
                      <span className="text-sm font-semibold min-w-[80px]">
                        Dataset:
                      </span>
                      <span className="text-sm text-muted-foreground">
                        {project.dataset}
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-sm font-semibold min-w-[80px]">
                        Outcome:
                      </span>
                      <span className="text-sm text-muted-foreground">
                        {project.keyOutcome}
                      </span>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-sm font-semibold min-w-[80px]">
                        Tools:
                      </span>
                      <div className="flex flex-wrap gap-2">
                        {project.tools.map((tool) => (
                          <span
                            key={tool}
                            className="px-2 py-1 text-xs bg-primary/10 text-primary rounded-full border border-primary/20"
                          >
                            {tool}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-3">
                    {project.links.project && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="border-primary text-primary hover:bg-primary hover:text-white"
                        asChild
                      >
                        <a
                          href={project.links.project}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <ExternalLink className="mr-2 h-4 w-4" />
                          View Project
                        </a>
                      </Button>
                    )}

                    {project.links.powerbi && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="border-secondary text-secondary hover:bg-secondary hover:text-white"
                        asChild
                      >
                        <a
                          href={project.links.powerbi}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <BarChart2 className="mr-2 h-4 w-4" />
                          Power BI
                        </a>
                      </Button>
                    )}

                    <Button
                      variant="outline"
                      size="sm"
                      className="border-muted-foreground/50 text-muted-foreground hover:bg-muted-foreground hover:text-white"
                      asChild
                    >
                      <a
                        href={project.links.github}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Github className="mr-2 h-4 w-4" />
                        GitHub
                      </a>
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Projects;
