<template>
<div v-if="agent">
<b-card>
        <div class="h4 text-center">
        <b-link :to="`/agent/view/${agent.ident}`">
        <div class="">{{ agent.ident }}</div>
        <b-img  style="max-height:160px;" :src="`/img/agent_spec_icons/agent_${getImageId(agent.specId)}.png`" alt="Image" class="rounded-0 mt-3" ></b-img> 
            
        </b-link>
        </div>

        <div class="text-center">

            <span class="">
                <b-link :to="`/agent_spec/${agent.specId}`">{{ agent.spec.displayedName }}</b-link>
            </span>
        </div>
</b-card>
</div>
</template>

<script >
export default {
    props: {
        agent: Object,
        
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
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style scoped lang="scss">

</style>
