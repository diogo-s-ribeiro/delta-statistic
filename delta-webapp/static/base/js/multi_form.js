const steps = Array.from(document.querySelectorAll('form .step'))
const nextBtn = document.querySelectorAll('form .next-btn');
const prevBtn = document.querySelectorAll('form .previous-btn');
const form = document.querySelectorAll('form');

nextBtn.forEach(button=>{
    button.addEventListener('click', () => {
        changeStep('next');
    })
})
prevBtn.forEach(button=>{
    button.addEventListener('click', () => {
        changeStep('prev');
    })
})

function changeStep(btn){
    let index = 0
    const active = document.querySelector('form .step.active');
    index = steps.indexOf(active);
    steps[index].classList.remove('active');
    if(btn === 'next'){
        index ++;
    }else if(btn === 'prev' ){
        index --;
    }
    steps[index].classList.add('active')
}