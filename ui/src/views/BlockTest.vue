<template>
<div>
    <h1><i class="fa fa-question-circle"></i> Blocktest</h1>
    <div id="geditor" class="svg-container"></div>

    <div id="cy" ></div>

</div>
</template>

<script>
import {
    SVG
} from '@svgdotjs/svg.js'

import '@svgdotjs/svg.draggable.js'
import cytoscape, {
    CytoscapeOptions,
    Core,
    EventNames,
    EventHandler
} from 'cytoscape'
// import {cytoscape} from '@cytoscapecytoscape/dist/cytoscape.min.js'
export default {
    name: 'Help',
    components: {},
    data() {
        return {

            //

        };
    },
    computed: {
        //

    },
    methods: {

        //

    },
    mounted() {

        var cy = cytoscape({

            container: document.getElementById('cy'), // container to render in

            elements: [ // list of graph elements to start with
                { // node a
                    data: {
                        id: 'a'
                    }
                },
                { // node b
                    data: {
                        id: 'b'
                    }
                },
                { // edge ab
                    data: {
                        id: 'ab',
                        source: 'a',
                        target: 'b'
                    }
                }
            ],

            style: [ // the stylesheet for the graph
                {
                    selector: 'node',
                    style: {
                        'background-color': '#666',
                        'label': 'data(id)'
                    }
                },

                {
                    selector: 'edge',
                    style: {
                        'width': 3,
                        'line-color': '#ccc',
                        'target-arrow-color': '#ccc',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }
                }
            ],

            layout: {
                name: 'grid',
                rows: 1
            }

        });

        
        var draw = SVG()
            .addTo('#geditor')
            .size(400, 400).attr({
                fill: '#fff',
                stroke: "#fff"
            })

        var part = draw.rect(30, 30)
        part.attr({
            fill: '#f6f'
        })

        var component = draw.group()
        var rect = draw.rect(100, 100)
        var partslot = draw.rect(30, 30).fill('#222').move(20, 20)
        var rect2 = draw.rect(50, 200)
        rect2.attr({
            fill: '#f06'
        })
        component.add(rect)
        component.add(rect2)
        component.add(partslot)

        part.draggable()
        part.on(['mouseover', 'click'], function () {
            this.front()
        })
        component.draggable()


        component.draggable().on('dragmove', e => {
            const {
                handler,
                box
            } = e.detail
            e.preventDefault()
            e.detail.handler.move(100, 200)

            let {
                x,
                y
            } = box
            handler.move(x, y)
            // events are still bound e.g. dragend will fire anyway
        })

        part.draggable().on('dragmove', e => {
            const {
                handler,
                box
            } = e.detail
            e.preventDefault()
            e.detail.handler.move(100, 200)

            let {
                x,
                y
            } = box

            handler.move(x, y)
        })

    }
}
</script>
