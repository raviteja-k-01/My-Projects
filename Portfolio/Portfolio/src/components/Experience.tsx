import { Briefcase, Building2 } from "lucide-react";

const Experience = () => {
  const experiences = [
    {
      title: "Data Analytics Trainee",
      company: "Codegnan IT Solutions",
      location: "Hyderabad",
      period: "Jul 2025 – Present",
      description: [
        "Processed and analyzed 10,000+ records using SQL and Python to derive performance metrics, uncover trends, and resolve data inconsistencies",
        "Developed MySQL queries (joins, aggregations, filters) and performed data cleaning, validation, and transformation to ensure data accuracy",
        "Created analytical summaries and visuals to highlight churn drivers and performance gaps",
      ],
      gradient: "from-primary/20 to-primary/5",
    },
    {
      title: "AI Intern",
      company: "SmartKnower",
      location: "Remote",
      period: "Nov 2022 – Dec 2022",
      description: [
        "Worked on image classification using the CIFAR-10 dataset, focusing on data preprocessing, normalization, and model evaluation",
        "Analyzed model performance using accuracy metrics, classification reports, and confusion matrices",
        "Created visualizations to communicate model behavior and performance insights to the team",
      ],
      gradient: "from-primary/20 to-primary/5",
    },
  ];

  return (
    <section id="experience" className="py-20 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-background via-secondary/5 to-background" />
      
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16 animate-fade-in">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Professional Experience
            </h2>
            <div className="w-20 h-1 bg-gradient-to-r from-primary to-secondary mx-auto mb-8 rounded-full" />
            <p className="text-base sm:text-lg text-muted-foreground">
              Building expertise through hands-on projects and real-world applications
            </p>
          </div>

          <div className="relative">
            {/* Timeline Line */}
            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary to-secondary hidden sm:block" />

            <div className="space-y-8">
              {experiences.map((exp, index) => (
                <div
                  key={exp.title}
                  className="glass-card p-6 sm:p-8 hover:shadow-xl transition-all duration-300 animate-slide-up relative sm:ml-20 group"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  {/* Timeline Dot */}
                  <div className="hidden sm:flex absolute -left-[5.5rem] top-8 w-12 h-12 rounded-full bg-gradient-to-br from-primary to-secondary items-center justify-center shadow-lg group-hover:scale-110 transition-transform">
                    <Briefcase className="h-6 w-6 text-white" />
                  </div>

                  <div className="flex items-start gap-4 mb-4">
                    <div className="sm:hidden w-12 h-12 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center">
                      <Briefcase className="h-6 w-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl sm:text-2xl font-bold text-foreground group-hover:text-primary transition-colors">
                        {exp.title}
                      </h3>
                      <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4 mt-2">
                        <div className="flex items-center gap-2 text-muted-foreground">
                          <Building2 className="h-4 w-4 text-secondary" />
                          <span className="font-semibold">{exp.company}</span>
                        </div>
                        <span className="hidden sm:inline text-muted-foreground">•</span>
                        <span className="text-muted-foreground">{exp.location}</span>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">{exp.period}</p>
                    </div>
                  </div>

                  <ul className="space-y-2 mt-4">
                    {exp.description.map((item, i) => (
                      <li key={i} className="flex items-start gap-3 text-sm sm:text-base text-muted-foreground">
                        <span className="text-primary mt-1 flex-shrink-0">▪</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Experience;
