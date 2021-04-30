<script>
import {
    Line,
    mixins,
    BaseChart,
    
} from 'vue-chartjs'

import zoom from 'chartjs-plugin-zoom';

const {
    reactiveProp
} = mixins

export default {
    extends: Line,
    mixins: [reactiveProp],
    props: ['chartData','options'],
    data(){
        return{
            graphKey:0
        }
    },
    methods:{
        computedChartData(){
            // return this.chartData
            // if (!this.chartData){
            //     return {}
            // }
            var ctx = this.$refs.canvas.getContext("2d")
            var gradient1 = ctx.createLinearGradient(0, 0, 0, 450);
            // gradient1.addColorStop(1, this.chartData.datasets[0].borderColor);   
            // gradient1.addColorStop(0.5, this.chartData.datasets[0].borderColor);   
            // gradient1.addColorStop(0, 'rgba(0,0,0,0)');
            gradient1.addColorStop(1, 'rgba(255, 0,0, 1)') // show this color at 0%;
            gradient1.addColorStop(0.5, 'rgba(255, 0, 0, 0.5)'); // show this color at 50%
            gradient1.addColorStop(0, 'rgba(255, 0, 0, 0)'); // show this color at 100%
            this.chartData.datasets[1].backgroundColor = gradient1

            var gradient2 = ctx.createLinearGradient(0, 0, 0, 450);
            gradient2.addColorStop(0, 'rgba(255, 0,0, 1)') // show this color at 0%;
            gradient2.addColorStop(0.5, 'rgba(255, 0, 0, 0.5)'); // show this color at 50%
            gradient2.addColorStop(1, 'rgba(255, 0, 0, 0)'); // show this color at 100%

             this.chartData.datasets[2].backgroundColor = gradient2
            return this.chartData;

        },
        computedOptions(){

            return this.options
        }
    },
    mounted() {

        // this.chartData is created in the mixin.
        // If you want to pass options please create a local options object
        this.addPlugin(zoom);
        this.renderChart(this.chartData, this.options)
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
