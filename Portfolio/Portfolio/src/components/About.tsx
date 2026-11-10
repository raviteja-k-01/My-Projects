import { GraduationCap, Target, Wrench } from "lucide-react";

const About = () => {
  const cards = [
    {
      icon: GraduationCap,
      title: "Education",
      content: "B.Tech in Artificial Intelligence and Data Science\nJBIET, Hyderabad\nCGPA: 7.4/10",
      gradient: "from-primary/20 to-primary/5",
    },
    {
      icon: Target,
      title: "Strengths",
      content: "Problem Solving\nAnalytical Thinking\nAgile Learning\nAttention to Detail",
      gradient: "from-secondary/20 to-secondary/5",
    },
    {
      icon: Wrench,
      title: "Tools I Use",
      content: "Python, SQL, Jupyter\nPower BI, Excel\nGitHub, VS Code\nPandas, NumPy",
      gradient: "from-primary/20 to-secondary/5",
    },
  ];

  return (
    <section className="py-20 relative overflow-hidden">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 animate-fade-in">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              About Me
            </h2>
            <div className="w-20 h-1 bg-gradient-to-r from-primary to-secondary mx-auto mb-8 rounded-full" />
            <div className="text-base sm:text-lg text-muted-foreground max-w-3xl mx-auto leading-relaxed space-y-4">
              <p>
                I am a Data Analyst with hands-on experience using SQL, Python, Excel, and Power BI to analyze real-world business datasets.
              </p>
              <p>
                I specialize in transforming raw data into actionable insights through exploratory analysis, KPI reporting, and interactive dashboards that support operational and business decisions.
              </p>
              <p className="font-medium">
                Currently seeking entry-level Data Analyst, Reporting Analyst, or BI Analyst roles where I can contribute analytical value from day one.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
            {cards.map((card, index) => {
              const Icon = card.icon;
              return (
                <div
                  key={card.title}
                  className="glass-card p-6 sm:p-8 hover:scale-105 transition-all duration-300 hover:shadow-xl animate-slide-up group"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${card.gradient} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                    <Icon className="h-7 w-7 text-primary" />
                  </div>
                  <h3 className="text-xl font-bold mb-4 text-foreground">
                    {card.title}
                  </h3>
                  <p className="text-muted-foreground whitespace-pre-line leading-relaxed">
                    {card.content}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
