"""Generate figures and evidence-based Markdown reports from actual runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evolution.common import save_csv,save_json
from evolution import routing as routing,scheduling as scheduling

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'figure.dpi':140,'font.size':10,'axes.grid':True,'grid.alpha':.2})
LABELS={'ga_small':'GA sigma=0.01L','ga_large':'GA sigma=0.05L','random':'Random search',
        'penalty_weak_inversion':'Penalty 0.05 / inversion','penalty_strong_inversion':'Penalty 100 / inversion',
        'penalty_strong_swap':'Penalty 100 / swap','nsga2':'NSGA-II','weighted':'Weighted sum',
        'qing_fixed':'Qing / fixed','qing_adaptive':'Qing / adaptive',
        'ellipsoid_fixed':'Ellipsoid / fixed','ellipsoid_adaptive':'Ellipsoid / adaptive'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def table(summary):
    lines=['| Метод | Лучшее | Среднее | Медиана | s | Худшее | Запусков |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for method,row in summary.items():
        lines.append('| '+method+' | '+' | '.join(f'{row[key]:.6g}' for key in ['best','mean','median','std','worst'])+f' | {row["runs"]} |')
    return '\n'.join(lines)


def flow(path,labels):
    fig,ax=plt.subplots(figsize=(10,1.7));ax.set_axis_off()
    for i,label in enumerate(labels):
        x=(i+.5)/len(labels)
        ax.text(x,.5,label,ha='center',va='center',transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.6',fc='#edf4fa',ec='#2374ab'),fontsize=9)
        if i<len(labels)-1:ax.annotate('',xy=((i+1.24)/len(labels),.5),xytext=((i+.78)/len(labels),.5),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#2374ab'})
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)


def convergence(histories,path,title,log=False,columns=(1,)):
    fig,ax=plt.subplots(figsize=(9,5))
    for method,runs in histories.items():
        # Step interpolation: no future improvement is shown before evaluation.
        end=min(run[-1][0] for run in runs);start=max(run[0][0] for run in runs)
        grid=np.linspace(start,end,200)
        values=[]
        for run in runs:
            a=np.array(run);idx=np.searchsorted(a[:,0],grid,side='right')-1
            values.append(a[idx,columns[0]])
        values=np.array(values)
        if log:values=np.maximum(values,1e-30)
        line,=ax.plot(grid,values.mean(axis=0),label=LABELS[method])
        ax.fill_between(grid,values.min(axis=0),values.max(axis=0),alpha=.12,color=line.get_color())
    if log:ax.set_yscale('log')
    ax.set(xlabel='Objective evaluations',ylabel=title,title='Mean and min–max across independent runs')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(path);plt.close(fig)


def generate(numbers,config):
    for number in numbers:
        folder=ROOT/f'lab{number}';out=folder/'results';figures=folder/'figures';figures.mkdir(exist_ok=True)
        summary=read(out/'summary.json');best=read(out/'best.json');hist=read(out/'histories.json');config=read(out/'config.json');cfg=config[f'lab{number}']
        flow(figures/'flow.png',['Initialisation','Evaluation','Parent selection','Variation','Elitist survival','Budget stop'])
        common=f'''\n## Экспериментальный протокол

Вариант 12: N=14, Q=1 (группа 517), Y=26; V=1+((17·14+7+26) mod 20)=12.
Seed данных: {config['data_seed']}. Seed запусков: {config['first_seed']}…{config['first_seed']+cfg['runs']-1}.
В каждой конфигурации {cfg['runs']} независимых запусков, бюджет ровно {cfg['budget']} вычислений.
Инициализация входит в бюджет. Повторная оценка элиты не производится.
Сравниваемые методы получают одинаковые seed, но различное потребление случайных чисел не означает попарного совпадения траекторий.
Время измерено perf_counter и зависит от машины; основной бюджет задан числом оценок.
Полные параметры: [results/config.json](results/config.json); каждый запуск: [results/runs.csv](results/runs.csv).
Траектории: [results/histories.json](results/histories.json), решения: [results/best.json](results/best.json).
Стандартное отклонение выборочное, ddof=1. Среднее не заменяет анализ допустимости.

![Эволюционный цикл](figures/flow.png)

## Результаты серии

{table(summary)}
'''
        if number==1:
            convergence(hist,figures/'convergence.png','Best Qing objective',True)
            small,large,random=[summary[k]['median'] for k in ['ga_small','ga_large','random']]
            text='''# Лабораторная работа 1 — многомерная оптимизация

## Постановка и представление

Минимизируется f(x)=Σᵢ₌₁⁵(xᵢ²−i)² на [-500,500]⁵. Размерность d=5+(12 mod 6)=5.
Генотип и фенотип — вещественный вектор из пяти координат; декодирование тождественное.
Точный глобальный минимум равен 0: xᵢ=±√i, всего 32 глобальных минимума.
Неотрицательность каждого слагаемого даёт нижнюю границу и доказывает этот ориентир.
В исходном документе формула и область заданы изображениями, при реализации проверены обе.

## Алгоритм и параметры

Популяция 60, равномерная инициализация; турнир размера 3; BLX-0.3 с вероятностью 0.9;
независимая гауссовская мутация каждой координаты с вероятностью 1/d=0.2;
σ=0.01L или 0.05L, где L=1000. Единственный изменяемый фактор — σ.
Выход за границы отражается с учётом любого количества пересечений. Одна лучшая особь сохраняется.
Остановка строго по бюджету; неполное последнее поколение допускается.
BLX — дополнительное задание. Он исследует окрестности родительского интервала и сохраняет вещественный генотип.
Random baseline равномерно выбирает точки из той же области при том же бюджете.

## Почему задача трудна

У функции несколько симметричных бассейнов притяжения; область очень велика относительно окрестности ±√i.
Равномерная выборка расходует почти все оценки вдали от хороших решений.
Локальный поиск может выбрать любой из равноценных бассейнов; сама разделимость делает локальную донастройку относительно простой.
Поэтому функция не является доказательством превосходства ГА над градиентными методами.
'''+common+f'''
![Сходимость](figures/convergence.png)

## Анализ и вывод

Медиана при малом σ: {small:.6g}, при большом: {large:.6g}; случайный поиск: {random:.6g}.
Отношение медиан большого и малого шага: {large/max(small,1e-300):.3g}.
Меньший шаг облегчает точную донастройку около минимума, большой продолжает выбрасывать координаты из малых бассейнов.
Показатели серии позволяют проверить этот эффект по всем запускам. Ноль известен аналитически;
остаточная ошибка прямо измеряет расстояние по целевой функции до глобального ориентира.
Результат — приближённое решение, гарантия достижения нуля не заявляется.
Использование фиксированного масштаба на огромной области ограничивает точность; развитие — уменьшающийся или адаптивный шаг.
'''
        elif number==2:
            data=read(folder/'data.json');winner=best['penalty_strong_inversion'];route=winner['solution']
            coords=np.array(data['coordinates']);cycle=[0,*route,0]
            fig,ax=plt.subplots(figsize=(7,6));ax.plot(coords[cycle,0],coords[cycle,1],'-o',lw=1)
            for i,(x,y) in enumerate(coords):ax.text(x+1,y+1,str(i))
            ax.set(xlabel='x, km',ylabel='y, km',title='Delivery tour; 0 = depot');fig.tight_layout();fig.savefig(figures/'route.png');plt.close(fig)
            convergence(hist,figures/'convergence.png','Distance of best feasible / least violating tour')
            rng=np.random.default_rng(config['data_seed']+1);examples=[]
            good=[route,data['reference_route']]
            for item in good:
                value=routing.objectives([item],data)[0]
                examples.append({'route':item,'distance':float(value[0]),'time':float(value[1]),'violation':float(value[2]),'feasible':bool(value[2]<1e-9)})
            for _ in range(10000):
                item=rng.permutation(np.arange(1,len(coords))).tolist();value=routing.objectives([item],data)[0]
                if value[2]>0:examples.append({'route':item,'distance':float(value[0]),'time':float(value[1]),'violation':float(value[2]),'feasible':False})
                if len(examples)==4:break
            if len(examples)!=4 or not all(e['feasible'] for e in examples[:2]):raise ValueError('Missing required feasibility examples')
            save_json(folder/'examples.json',examples)
            details=[];arrival=0.
            for a,b in zip(cycle,cycle[1:]):
                arrival+=data['time'][a][b]
                details.append({'from':a,'to':b,'distance_km':data['distance'][a][b],'travel_minutes':data['time'][a][b],'arrival_minutes':arrival,'service_minutes':data['service'][b]})
                arrival+=data['service'][b]
            save_csv(out/'delivery.csv',details)
            text=f'''# Лабораторная работа 2 — маршрут доставки

## Предметная модель и ограничения

20 клиентов и депо 0. Замкнутый маршрут посещает каждого клиента ровно один раз и возвращается в депо.
Минимизируется сумма евклидовых расстояний. Ограничение: сумма времени на дугах и обслуживания ≤ T={data['time_limit']:.3f} мин.
Все точки обязательны; удаление клиентов для выполнения лимита запрещено.
Генотип — перестановка 1…20; декодирование добавляет депо в начало и конец.
Координаты в квадрате 100×100 км, seed={config['data_seed']}; дуговые множители трафика U[0.7,2.8],
скорость 35 км/ч, обслуживание от 3 до 8 минут. Трафик асимметричен.
T равен 1.12 времени конструктивного маршрута ближайшего соседа по расстоянию.
Это гарантирует наличие допустимого решения, но не гарантирует допустимость оптимального по длине маршрута.
Полные матрицы сохранены в [data.json](data.json); внешние данные не используются.

## Операторы и сравнения

Популяция 60, турнир 3, OX с вероятностью 0.9, мутация с вероятностью 0.8, элита 1.
OX сохраняет сегмент первого родителя и относительный порядок остальных точек второго.
Инверсия сегмента и обмен двух точек сохраняют перестановку, поэтому структурный ремонт не нужен.
Временное ограничение учитывается штрафом: score=distance+λ·max(0,time−T).
Сравнение λ=0.05 и λ=100 при одной инверсии меняет только обработку ограничений;
сравнение инверсии и обмена при λ=100 меняет только оператор.
Random baseline использует тот же бюджет оценок случайных перестановок; из допустимых сохраняется кратчайшая.
Отдельный внешний регистратор лучшего решения у всех методов использует сначала допустимость,
затем минимальное нарушение или длину. Это не влияет на отбор родителей.
Два допустимых и два недопустимых примера с полными перестановками: [examples.json](examples.json).
'''+common+'\n| Метод | Доля допустимых итогов |\n|---|---:|\n'+'\n'.join(f'| {k} | {v["feasible_fraction"]:.0%} |' for k,v in summary.items())+f'''

![Маршрут](figures/route.png)
![Сходимость](figures/convergence.png)

## Предметный результат и вывод

Маршрут: 0 → {' → '.join(map(str,route))} → 0.
Длина {winner['value']:.3f} км, время {winner['duration']:.3f} мин, превышение {winner['violation']:.6g} мин.
Пошаговое время прибытия и обслуживание: [delivery.csv](results/delivery.csv).
Ближайший сосед: {examples[1]['distance']:.3f} км и {examples[1]['time']:.3f} мин.
Он является дополнительным ориентиром, его построение не входит в бюджеты сравниваемых поисков.
Средние длины в строке random включают недопустимые итоги: их нельзя считать качеством допустимых маршрутов.
Малый штраф дал медиану {summary['penalty_weak_inversion']['median']:.3f} км,
большой с инверсией — {summary['penalty_strong_inversion']['median']:.3f} км:
сильное давление на допустимость здесь ухудшило типичную длину, хотя оба ГА нашли допустимые решения во всех запусках.
Инверсия меняет порядок большого сегмента и при асимметричном трафике способна существенно изменить время;
обмен изменяет локальные связи. Их эффект виден в медианах и долях допустимости.
Штраф 0.05 допускает обмен больших временных нарушений на небольшое сокращение расстояния;
100 значительно сильнее направляет отбор в допустимую область.
Приемлемость результата определяется одновременно выполнением T и длиной относительно конструктивного ориентира.
Без точного решения нельзя утверждать глобальную оптимальность; ограничены модель трафика и один экземпляр данных.
'''
        elif number==3:
            data=read(folder/'data.json');ns=best['nsga2'];front=np.array(ns['front']);solutions=np.array(ns['solutions'])
            fig,axes=plt.subplots(1,3,figsize=(12,4))
            for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
                for method,color in [('weighted','#ee7733'),('nsga2','#0077bb')]:
                    ff=np.unique(best[method]['front'],axis=0);ax.scatter(ff[:,a],ff[:,b],label=LABELS[method],alpha=.6,c=color)
                names=['Conflicts','Student gaps','Late events'];ax.set(xlabel=names[a],ylabel=names[b]);ax.legend(fontsize=8)
            fig.tight_layout();fig.savefig(figures/'pareto.png');plt.close(fig)
            # Three different objective vectors, chosen greedily for separation.
            unique,idx=np.unique(front,axis=0,return_index=True);norm=unique/np.maximum(np.ptp(unique,axis=0),1)
            selected=[int(np.lexsort((unique[:,1],unique[:,2],unique[:,0]))[0])]
            second=int(np.lexsort((unique[:,1],unique[:,0],unique[:,2]))[0])
            if second not in selected:selected.append(second)
            while len(selected)<min(3,len(unique)):
                distances=np.min(np.linalg.norm(norm[:,None,:]-norm[selected][None,:,:],axis=2),axis=1);distances[selected]=-1;selected.append(int(distances.argmax()))
            reps=[]
            for i in selected:
                schedule=solutions[idx[i]];reps.append({'objectives':unique[i].tolist(),'slots':schedule.tolist()})
            save_json(out/'representatives.json',reps)
            events=[]
            for i,slot in enumerate(reps[0]['slots']):events.append({'event':i,'cohort':data['cohorts'][i],'teacher':data['teachers'][i],'day':slot//5+1,'lesson':slot%5+1})
            save_csv(out/'timetable.csv',events)
            text=f'''# Лабораторная работа 3 — многокритериальное расписание

## Трек A и формальная модель

24 занятия, 6 групп студентов, 5 преподавателей, 5 дней по 5 временных слотов.
Генотип — вектор 24 целочисленных слотов из доступных множеств внутри 0…24; фенотип — таблица «занятие — день — пара».
Пара конфликтует, если имеет общую группу или преподавателя; таких пар {len(data['conflicts'])}.
Генерация: группа i mod 6, преподаватель равномерно 0…4; seed={config['data_seed']}.
У каждой группы основной день (номер группы mod 5); каждому её занятию доступны
первая, третья и четвёртая пара этого дня и пятая пара следующего дня по циклу.
Это жёсткое ограничение доступности: инициализация и сброс гена выбирают только разрешённые слоты,
а кроссовер наследует допустимый слот того же занятия. Два ранних слота для четырёх занятий
создают конфликт между отсутствием пересечений и отказом от поздних пар.
Ограничений аудиторий в этой модели нет: число доступных аудиторий не ограничено.
Данные: [data.json](data.json). Все генотипы структурно допустимы; конфликтное расписание организационно недопустимо.

Три минимизируемые цели: f₁ — число конфликтующих пар в одном слоте;
f₂ — число незанятых слотов между первой и последней парой в каждом дне каждой группы;
f₃ — число занятий на четвёртой и пятой парах дня.
Пересечение занятий штрафуется f₁, а не исключается жёстким ремонтом: это позволяет изучать компромиссы.
Для практического выбора из фронта необходимо требовать f₁=0.

## Архитектура NSGA-II

Реализованы строгое доминирование, полная недоминируемая сортировка, crowding distance,
бинарный турнир по рангу и расстоянию, отбор из объединения родителей и потомков.
Популяция 60; равномерный кроссовер каждого гена; вероятность сброса гена в случайный слот 1/24.
Потомки всегда имеют корректный диапазон. Новые механизмы — Парето-сортировка и поддержание разнообразия.
Внешний архив хранит недоминируемые решения; для каждого одинакового вектора целей остаётся один генотип-свидетель.

Baseline — турнирный ГА со взвешенной суммой 0.5·f₁/C+0.3·f₂/96+0.2·f₃/24,
где C — число конфликтных пар. Нормировка фиксирована и не зависит от текущей популяции.
Архив одинаков у обоих методов; оценка не даёт NSGA-II преимущества за счёт хранения решений.
Сравнивается точный трёхмерный гиперобъём архивов относительно общей точки (C+1,97,25), больше — лучше.
Время вычисления сортировки и архивирования приведено в CSV, бюджет фитнес одинаков.
'''+common+'\n| Метод | Средний HV | s(HV) | Среднее число разных векторов целей | Доля запусков с f₁=0 |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f'| {k} | {v["hypervolume_mean"]:.3f} | {v["hypervolume_std"]:.3f} | {v["front_size_mean"]:.2f} | {v["feasible_fraction"]:.0%} |' for k,v in summary.items())+f'''

В общей таблице value — минимальная взвешенная сумма в архиве. Она вспомогательная;
основная метрика покрытия компромиссов — HV. Гиперобъём зависит от выбранной общей опорной точки.

![Проекции фронта](figures/pareto.png)

## Характерные решения

Векторы (конфликты, окна, поздние занятия): {', '.join(str(rep['objectives']) for rep in reps)}.
Первое минимизирует конфликты (при равенстве — поздние занятия), второе — поздние занятия;
третье выбирается для покрытия удалённой части фронта.
Меньшее первое число означает более пригодное расписание, второе — удобство студентов, третье — отказ от поздних пар.
Это компромиссы выбранного запуска, а не доказанный точный фронт всей задачи.
Полные три вектора назначений: [representatives.json](results/representatives.json);
таблица первого расписания: [timetable.csv](results/timetable.csv).

## Вывод и ограничения

Взвешенная сумма задаёт одно предпочтение; NSGA-II поддерживает несколько направлений улучшения одновременно.
Разница средних HV и числа уникальных векторов показывает влияние поддержания разнообразия.
Средний HV NSGA-II: {summary['nsga2']['hypervolume_mean']:.3f}; взвешенная сумма: {summary['weighted']['hypervolume_mean']:.3f}.
Доля запусков с бесконфликтным решением: {summary['nsga2']['feasible_fraction']:.0%} и {summary['weighted']['feasible_fraction']:.0%} соответственно.
Положительное f₁ означает, что решение нельзя применять без исправления конфликтов,
даже если его окна и поздние занятия лучше. Качество нужно оценивать по всем трём координатам.
Модель не включает вместимость аудиторий, индивидуальные предпочтения и переносы между корпусами.
Увеличение бюджета и включение специализированной мутации конфликтных занятий могут улучшить применимость.
'''
        else:
            convergence({k:v for k,v in hist.items() if k.startswith('qing')},figures/'qing.png','Best Qing objective',True)
            convergence({k:v for k,v in hist.items() if k.startswith('ellipsoid')},figures/'ellipsoid.png','Best ellipsoid objective',True)
            steps=read(out/'steps.json');fig,ax=plt.subplots(figsize=(9,5))
            a=np.array(steps['ellipsoid_adaptive']);median=np.median(a[:,:,1:],axis=0)
            for i in range(5):ax.plot(a[0,:,0],median[:,i],label=f'Coordinate {i+1}')
            ax.set(yscale='log',xlabel='Objective evaluations',ylabel='Median inherited sigma',title='Across 30 runs: median of population coordinate medians');ax.legend();fig.tight_layout();fig.savefig(figures/'steps.png');plt.close(fig)
            pairs=[]
            for func in ['qing','ellipsoid']:
                fixed=np.array([run[-1][1] for run in hist[func+'_fixed']]);adaptive=np.array([run[-1][1] for run in hist[func+'_adaptive']])
                wins=int(np.sum(adaptive<fixed));ties=int(np.sum(adaptive==fixed))
                pairs.append({'function':func,'adaptive_wins':wins,'ties':ties,'fixed_wins':len(fixed)-wins-ties,'median_ratio_fixed_to_adaptive':float(np.median(fixed)/max(np.median(adaptive),1e-300))})
            save_json(out/'paired_comparison.json',pairs)
            text='''# Лабораторная работа 4 — самоадаптирующаяся стратегия эволюции

## Исследовательский вопрос

Улучшает ли наследуемый покоординатный шаг мутации точность (μ+λ)-ES при одном бюджете
относительно фиксированного шага на мультимодальной и анизотропной функции?
Заявка и риски: [PROPOSAL.md](PROPOSAL.md). Заявка документирует тему; факт утверждения преподавателем не заявляется.

## Модель и архитектура

Две функции размерности 5 на [-10,10]⁵: Цин Σ(xᵢ²−i)² и эллипсоид Σ10^(4(i−1)/4)xᵢ².
У обеих минимум 0; у эллипсоида x=0, у Цин xᵢ=±√i.
Диапазон Цин здесь уменьшен относительно ЛР1 специально для исследования донастройки;
сравнивать абсолютные результаты ЛР1 и ЛР4 как одинаковые условия нельзя.
Число обусловленности эллипсоида 10⁴; разные координаты требуют разных шагов.
Данные аналитические, внешних источников и симуляционных случайностей нет.

Генотип адаптивной ES: (x₁…x₅,σ₁…σ₅), фенотип x. Начальные σ=1 для обеих версий.
μ=40, λ=80; родители равновероятно выбираются с возвращением.
Рекомбинация x — арифметическое среднее, σ — геометрическое.
Адаптация: σᵢ′=σᵢ·exp(τ₀Z₀+τZᵢ), где τ₀=1/√(2d), τ=1/√(2√d),
Z₀ общий для особи, Zᵢ независимы. σ ограничен [10⁻¹⁰,20].
Затем xᵢ′=xᵢ+σᵢ′Nᵢ, границы отражаются; выбираются μ лучших из родителей и потомков.
В фиксированной версии наследуемые σ остаются равными 1. Это единственная разница алгоритмов.
Отбор плюс сохраняет найденный минимум; последнее поколение обрезается до бюджета.
Система состоит из генератора кандидатов, наследования стратегии, счётчика оценок,
элитистского отбора, журнала координатных шагов и экспериментального анализатора.
'''+common+'''
![Цин](figures/qing.png)
![Эллипсоид](figures/ellipsoid.png)
![Адаптация шагов](figures/steps.png)

## Устойчивость и эффект адаптации

'''+ '\n'.join(f'На {p["function"]}: адаптивная версия лучше в {p["adaptive_wins"]} из 30 пар, фиксированная в {p["fixed_wins"]}, совпадений {p["ties"]}; отношение медиан fixed/adaptive = {p["median_ratio_fixed_to_adaptive"]:.4g}.' for p in pairs)+'''

Сравнение попарно по seed является описательным; статистическая значимость без отдельного теста не утверждается.
Графики показывают среднее и полный min–max диапазон всех 30 запусков, а не одну удачную траекторию.
На эллипсоиде различие наследуемых σ по координатам показывает механизм приспособления к анизотропии.
Пригодность оценивается близостью к известному нулю, медианой и худшим запуском при равной стоимости фитнеса.

## Ограничения и развитие

Покоординатная адаптация не обучает поворот системы координат; повёрнутый эллипсоид потребовал бы ковариационной адаптации.
Исследованы две функции, одна размерность и один начальный масштаб. Эти результаты не доказывают универсальное превосходство.
На мультимодальной Цин возможна преждевременная сходимость к одному бассейну, хотя его глобальный минимум равноценен остальным.
Следующие эксперименты: вращение эллипсоида, размерности 10/20, рестарты и ковариационная стратегия.
План демонстрации на 5–7 минут: [DEMO.md](DEMO.md).
'''
        (folder/'REPORT.md').write_text(text,encoding='utf-8')
        (folder/'README.md').write_text(f'''# Лабораторная работа {number}

Вариант 12, группа 517, номер 14. Запуск из корня репозитория:

```console
python -m pip install -r requirements.txt
python run.py --lab {number}
```

Полная серия: {cfg['runs']} запусков на конфигурацию, бюджет {cfg['budget']} оценок на запуск.
Параметры берутся из `config.json`; `--seed 13000` меняет начало серии.
`--runs 2` запускает короткую проверку и перезаписывает результаты выбранной работы.
Для возвращения полной серии повторите команду без `--runs`.

[Отчёт](REPORT.md) · [Параметры](results/config.json) · [Таблица запусков](results/runs.csv)
''',encoding='utf-8')
