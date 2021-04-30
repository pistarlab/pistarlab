<template>
<div>
    <canvas id="chart"></canvas>
</div>
</template>

<script>
import Chart from 'chart.js'

export default {
    props: ['chartData', 'options'],
    compute: {
        computedOptions() {
            return this.options
        }
    },
    mounted() {

        const ctx = document.getElementById('chart');
        new Chart(ctx, {
            type: "line",
            data: this.chartData,
            options: this.options
        });
    
    // this.chartData is created in the mixin.
    // If you want to pass options please create a local options object
    // this.renderChart(this.chartData, this.computedOptions)
    //         this.renderChart({
    //       labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July','January', 'February', 'March', 'April', 'May', 'June', 'July','January', 'February', 'March', 'April', 'May', 'June', 'July','January', 'February', 'March', 'April', 'May', 'June', 'July','January', 'February', 'March', 'April', 'May', 'June', 'July'],
    //       datasets: [
    //         {
    //           label: 'Data One',
    //           backgroundColor: '#f87979',
    //           data: [40, 39, 10, 40, 39, 80, 40]
    //         }
    //       ]
    //     }, {responsive: true, maintainAspectRatio: false})

    //   }
}
}
</script>

<style>
</style>
