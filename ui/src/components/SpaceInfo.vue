<template>
<span>

    <span v-if="space.class_name == 'Box'">
        <span v-if="space.kwargs.shape.values.length == 3 
                && space.kwargs.shape.values[0] > space.kwargs.shape.values[2] 
                && space.kwargs.shape.values[1] > space.kwargs.shape.values[2] 
                && space.kwargs.shape.values[2]==3">
            <i class="fa fa-image"></i> Color Image: {{space.kwargs.shape.values[0]}} x {{space.kwargs.shape.values[1]}}
        </span>
        <span v-else>
            <i class="fa fa-square"></i> Box: {{space.kwargs.shape.values.length}}D - {{space.kwargs.shape.values.join("x")}}
        </span>
    </span>
    <span v-else-if="space.class_name == 'Discrete'">
        <i class="fa fa-grip-vertical"></i> {{space.class_name}}: {{space.args[0]}}
    </span>
    <span v-else>
        {{space.class_name}} (args={{space.args}},{{space.kwargs}})
    </span>

</span>
</template>

<script>
export default {
    props: {
        space: Object
    },
    data() {
        return {

        };
    },
    mounted() {
        //
    },
    methods: {
        //

    },
    computed: {
        //

    },
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<style lang="scss" scoped>

</style>
